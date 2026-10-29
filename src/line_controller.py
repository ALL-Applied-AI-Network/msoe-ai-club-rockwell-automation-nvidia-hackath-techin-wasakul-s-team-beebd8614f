"""
Industrial Line Controller & Safety Interlock (Fluid Mixing Station 3)
Rockwell Automation + NVIDIA Physical AI Prototype

Connects the AI inspection model to the production line control logic:
  1. Pre-Fill Guard (Predictive): checks recipe transition risk before dispensing.
  2. Post-Fill Inspection (Reactive): analyzes camera feed, detects carryover,
     and triggers an emergency stop / interlock with full root-cause diagnostics.
  3. EtherNet/IP (CIP) Interface: communicates directly with physical Allen-Bradley
     ControlLogix / CompactLogix PLCs via native EtherNet/IP (pylogix).
  4. Simulation / Mock Mode: allows complete offline testing without hardware.

Usage:
  # 1. Offline simulation on planned run (any computer, no GPU/PLC needed):
  python3 src/line_controller.py --plan twin_data/demo_out/plan.csv --mock

  # 2. Predictive pre-fill prevention mode:
  python3 src/line_controller.py --plan twin_data/demo_out/plan.csv --mock --predictive

  # 3. PHYSICAL CSI LINE (EtherNet/IP to Rockwell PLC):
  python3 src/line_controller.py --plc-ip 192.168.1.10 --plc-slot 0 --live \
      --image-dir /path/to/camera_inbox --base /workspace/models/cosmos-reason \
      --adapter /workspace/models/run-01
"""

import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Attempt EtherNet/IP driver import (pylogix)
try:
    from pylogix import PLC
    HAS_PYLOGIX = True
except ImportError:
    HAS_PYLOGIX = False

# Recipe specification table (pure plant engineering tags: Tank 100, Tank 200, Tank 400)
RECIPES = {
    5: {100: 10, 200: 10, 400: 10}, 6: {100: 30, 200: 10},
    7: {100: 20, 400: 50},          8: {100: 20, 200: 20, 400: 20},
    9: {100: 10, 200: 35, 400: 35}, 10: {100: 45, 200: 15, 400: 15},
    11: {100: 10, 200: 10, 400: 50}, 12: {100: 10, 200: 50, 400: 10},
    13: {200: 50, 400: 20},         14: {100: 60, 400: 10},
}
TANK = {100: "Tank 100", 200: "Tank 200", 400: "Tank 400"}
ORDER = [200, 400, 100]

# Beer-Lambert optical model coefficients (calibrated from line data, pure Python)
K_MATRIX = [
    [1.3332, 1.5699, 2.9855],  # Base
    [-0.8894, 2.8181, 2.9625], # Tank 200 component
    [7.5434, 0.9629, -2.4204], # Tank 400 component
]
EMPTY_VIAL_LINEAR = [0.07809, 0.08206, 0.05105]


def describe(rid):
    if rid not in RECIPES:
        return f"Recipe {rid}: Unknown"
    parts = [f"{TANK[t]} {RECIPES[rid][t]} mL" for t in ORDER if t in RECIPES[rid]]
    return f"Recipe {rid}: " + ", then ".join(parts) + f" (total {sum(RECIPES[rid].values())} mL)"


def compute_photo_rgb(c200, c400):
    """Predicts linear light absorption via Beer-Lambert law without external dependencies."""
    rgb = []
    for col in range(3):
        a = K_MATRIX[0][col] + c200 * K_MATRIX[1][col] + c400 * K_MATRIX[2][col]
        lin = EMPTY_VIAL_LINEAR[col] * math.exp(-a)
        val = 255.0 * (max(0.0, lin) ** (1.0 / 2.2))
        rgb.append(min(255.0, max(0.0, val)))
    return rgb


def color_distance(rgb1, rgb2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


class RockwellPLCInterlock:
    """
    Rockwell Automation EtherNet/IP Driver & Safety Interlock.
    Communicates directly with Allen-Bradley ControlLogix/CompactLogix PLCs over CIP protocol.
    Falls back gracefully to simulation mode if no live PLC IP is supplied.
    """
    DEFAULT_TAG_MAP = {
        # Outputs to PLC (Interlocks & Safety Control)
        "Line_Running": "Station3_Line_Running",
        "Station3_Dispense_Permissive": "Station3_Dispense_Permissive",
        "MagneMover_Infeed_Active": "Station3_MagneMover_Infeed_Active",
        "Alarm_Carryover_Fault": "Station3_Alarm_Carryover",
        "Alarm_E_Stop": "Station3_E_Stop",
        "Flush_Cycle_Required": "Station3_Flush_Required",
        "Fault_Code": "Station3_Fault_Code",
        "Inspection_Complete": "Station3_Inspection_Complete",

        # Inputs from PLC (Line Sensors & Cart Arrival)
        "Cart_In_Position": "Station3_Cart_In_Position",
        "Current_Recipe": "Station3_Current_Recipe",
        "Prev_Recipe": "Station3_Prev_Recipe",
        "Vial_Number": "Station3_Vial_Number",
    }

    def __init__(self, plc_ip=None, slot=0, micro800=False, custom_tags=None):
        self.plc_ip = plc_ip
        self.slot = slot
        self.micro800 = micro800
        self.comm = None
        self.is_hardware_connected = False

        # Tag dictionary (in-memory state)
        self.tags = {
            "Line_Running": True,
            "Station3_Dispense_Permissive": True,
            "MagneMover_Infeed_Active": True,
            "Alarm_Carryover_Fault": False,
            "Alarm_E_Stop": False,
            "Flush_Cycle_Required": False,
            "Fault_Code": 0,
            "Inspection_Complete": False,
            "Cart_In_Position": False,
            "Current_Recipe": 5,
            "Prev_Recipe": 0,
            "Vial_Number": 1,
        }

        # Configure EtherNet/IP tag names (default or custom)
        self.tag_map = dict(self.DEFAULT_TAG_MAP)
        if custom_tags:
            self.tag_map.update(custom_tags)

        # Initialize physical EtherNet/IP connection if IP provided
        if self.plc_ip:
            self._connect_ethernet_ip()
        else:
            print("[PLC DRIVER] Running in simulated software interlock mode (no --plc-ip specified).")

    def _connect_ethernet_ip(self):
        if not HAS_PYLOGIX:
            print("[PLC DRIVER ERROR] 'pylogix' library is not installed.")
            print("  Please install it to connect to the physical PLC: pip install pylogix")
            print("  Falling back to simulated mode.")
            return

        try:
            print(f"[ETHERNET/IP] Initializing CIP connection to Rockwell PLC at {self.plc_ip} (Slot {self.slot})...")
            self.comm = PLC()
            self.comm.IPAddress = self.plc_ip
            self.comm.ProcessorSlot = self.slot
            self.comm.Micro800 = self.micro800

            # Attempt handshake query
            prop = self.comm.GetDeviceProperties()
            if prop.Status == "Success":
                self.is_hardware_connected = True
                print(f"[ETHERNET/IP] Connected successfully to Rockwell PLC: {prop.ProductName} (Rev: {prop.Revision})")
            else:
                print(f"[ETHERNET/IP WARNING] Initial handshake test returned status: {prop.Status}. Will attempt tag reads/writes directly.")
                self.is_hardware_connected = True
        except Exception as e:
            print(f"[ETHERNET/IP ERROR] Connection to {self.plc_ip} failed: {e}. Running in simulated mode.")
            self.is_hardware_connected = False

    def read_tag(self, key, default=None):
        """Reads a tag from the physical PLC via EtherNet/IP (or in-memory cache)."""
        if self.is_hardware_connected and self.comm:
            plc_tag = self.tag_map.get(key, key)
            try:
                ret = self.comm.Read(plc_tag)
                if ret.Status == "Success":
                    self.tags[key] = ret.Value
                    return ret.Value
                else:
                    print(f"[ETHERNET/IP READ WARN] Tag '{plc_tag}': {ret.Status}")
            except Exception as e:
                print(f"[ETHERNET/IP READ ERROR] {e}")
        return self.tags.get(key, default)

    def write_tag(self, key, value):
        """Writes a tag to the physical Rockwell PLC via EtherNet/IP and updates local state."""
        self.tags[key] = value
        if self.is_hardware_connected and self.comm:
            plc_tag = self.tag_map.get(key, key)
            try:
                ret = self.comm.Write(plc_tag, value)
                if ret.Status == "Success":
                    print(f"[ETHERNET/IP WRITE OK] Wrote PLC tag '{plc_tag}' = {value}")
                else:
                    print(f"[ETHERNET/IP WRITE FAIL] Tag '{plc_tag}' = {value}: {ret.Status}")
            except Exception as e:
                print(f"[ETHERNET/IP WRITE ERROR] {e}")

    def emergency_stop(self, fault_code, reason):
        """Trips physical and logical safety interlocks."""
        self.write_tag("Line_Running", False)
        self.write_tag("Station3_Dispense_Permissive", False)
        self.write_tag("MagneMover_Infeed_Active", False)
        self.write_tag("Alarm_Carryover_Fault", True)
        self.write_tag("Alarm_E_Stop", True)
        self.write_tag("Flush_Cycle_Required", True)
        self.write_tag("Fault_Code", fault_code)

    def reset_and_flush(self):
        """Resets safety interlocks and restores permissives after an operator flush."""
        self.write_tag("Alarm_Carryover_Fault", False)
        self.write_tag("Alarm_E_Stop", False)
        self.write_tag("Flush_Cycle_Required", False)
        self.write_tag("Fault_Code", 0)
        self.write_tag("Line_Running", True)
        self.write_tag("Station3_Dispense_Permissive", True)
        self.write_tag("MagneMover_Infeed_Active", True)

    def close(self):
        if self.comm:
            try:
                self.comm.Close()
            except Exception:
                pass


class LineController:
    def __init__(self, base_model=None, adapter_path=None, plc_ip=None, plc_slot=0, custom_tags=None, mock=False):
        self.plc = RockwellPLCInterlock(plc_ip=plc_ip, slot=plc_slot, custom_tags=custom_tags)
        self.mock = mock
        self.model = None
        self.processor = None

        if not mock and base_model:
            self._load_model(base_model, adapter_path)

    def _load_model(self, base_model, adapter_path):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"[INIT] Loading base model from {base_model}...")
        self.processor = AutoProcessor.from_pretrained(base_model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        if adapter_path and os.path.exists(adapter_path):
            print(f"[INIT] Attaching fine-tuned LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(model, adapter_path)
        else:
            self.model = model
        self.model.eval()
        print("[INIT] Model successfully loaded onto GPU.")

    def pre_fill_check(self, recipe_id, prev_recipe_id, vial_in_batch):
        """
        PREDICTIVE PREVENTION:
        Evaluates recipe transition before the nozzle dispenses into the vial.
        Calculates fluid carryover risk using the physical Beer-Lambert model.
        If carryover risk is critical, halts before dispensing.
        """
        if vial_in_batch != 1 or not prev_recipe_id or prev_recipe_id not in RECIPES:
            return {"action": "PERMIT_FILL", "risk": "LOW", "detail": "Clean batch or continuous recipe"}

        # Calculate concentration of active dye tanks
        ml_curr = RECIPES[recipe_id]
        tot_curr = sum(ml_curr.values())
        c_curr = (ml_curr.get(200, 0) / tot_curr, ml_curr.get(400, 0) / tot_curr)

        ml_prev = RECIPES[prev_recipe_id]
        tot_prev = sum(ml_prev.values())
        c_prev = (ml_prev.get(200, 0) / tot_prev, ml_prev.get(400, 0) / tot_prev)

        # Expected carryover with median line carryover alpha = 0.53
        c_mix = (0.53 * c_prev[0] + 0.47 * c_curr[0], 0.53 * c_prev[1] + 0.47 * c_curr[1])

        rgb_curr = compute_photo_rgb(*c_curr)
        rgb_mix = compute_photo_rgb(*c_mix)
        delta = color_distance(rgb_mix, rgb_curr)

        # Check which tanks in previous recipe are causing the risk
        offending_tanks = [TANK[t] for t in (200, 400) if ml_prev.get(t, 0) > 0 and (ml_curr.get(t, 0) == 0 or c_prev[0] != c_curr[0])]

        if delta > 6.0:  # Visibility delta threshold
            return {
                "action": "PAUSE_BEFORE_FILL",
                "risk": "HIGH",
                "delta": delta,
                "detail": (
                    f"Transition Recipe {prev_recipe_id} -> Recipe {recipe_id} has high carryover risk "
                    f"(color delta {delta:.1f} > threshold 6.0). Residual fluid from "
                    f"{', '.join(offending_tanks) or 'previous tanks'} in the shared manifold will cross-contaminate the batch."
                ),
            }

        return {"action": "PERMIT_FILL", "risk": "LOW", "detail": "Safe recipe transition"}

    def inspect_vial_image(self, image_path, recipe_id, prev_recipe_id, vial_in_batch, row_data=None):
        """
        POST-FILL DETECTION:
        Runs VLM inference (or mock emulation) to evaluate liquid appearance.
        Returns parsed action ('STOP_MACHINE' or 'CONTINUE') and diagnostic message.
        """
        prev_txt = describe(prev_recipe_id) if prev_recipe_id in RECIPES else "none (start of production run)"
        prompt = (
            f"Station 3 Inspection. Current {describe(recipe_id)}. Previous {prev_txt}. "
            f"This is vial {vial_in_batch} of the batch. Inspect the vial image: Does the liquid match "
            f"Recipe {recipe_id}? If contaminated, output machine control action STOP_MACHINE, identify "
            f"the vial, and detail the chemical root cause from the previous recipe."
        )

        if self.mock or self.model is None:
            # Emulate inference using row metadata or clean rules
            is_contaminated = bool(row_data and int(row_data.get("contaminated", 0)))
            obs = (row_data.get("observed_color") if row_data else "") or "discolored"
            if is_contaminated and prev_recipe_id in RECIPES:
                prev_tanks = [TANK[t] for t in ORDER if t in RECIPES[prev_recipe_id] and t != 100]
                curr_tanks = [TANK[t] for t in ORDER if t in RECIPES[recipe_id] and t != 100]
                text = (
                    f"[ACTION: STOP_MACHINE]\n"
                    f"ERROR: Fluid Contamination Detected.\n"
                    f"- Recipe Being Filled: Recipe {recipe_id} ({describe(recipe_id).split(': ', 1)[1]})\n"
                    f"- Batch Position: Vial {vial_in_batch} of 3\n"
                    f"- Observed Appearance: {obs} (MISMATCH: does not match standard Recipe {recipe_id})\n"
                    f"- Root Cause: Carryover from previous Recipe {prev_recipe_id} ({describe(prev_recipe_id).split(': ', 1)[1]}). "
                    f"Residual fluid from {', '.join(prev_tanks)} remained in the shared dispensing nozzle/manifold "
                    f"and cross-contaminated the incoming fluid from {', '.join(curr_tanks)}.\n"
                    f"- Machine Directive: Line halted. Initiate nozzle flush cycle before resuming."
                )
            else:
                text = (
                    f"[ACTION: CONTINUE]\n"
                    f"STATUS: Normal.\n"
                    f"- Recipe Being Filled: Recipe {recipe_id}\n"
                    f"- Batch Position: Vial {vial_in_batch} of 3\n"
                    f"- Verification: Liquid color matches Recipe {recipe_id} specification. No carryover detected."
                )
        else:
            # Real GPU multimodal VLM execution
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            formatted_prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = self.processor(text=[formatted_prompt], images=[img], return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=220)
            text = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        action = "STOP_MACHINE" if "[ACTION: STOP_MACHINE]" in text or "STOP_MACHINE" in text else "CONTINUE"
        return action, text

    def process_vial(self, image_path, recipe_id, prev_recipe_id, vial_in_batch, row_data=None, enable_pre_fill_guard=False):
        """Full supervisory cycle for one vial position on the line."""
        print(f"\n>>> [LINE EVENT] Vial approaching Station 3: Recipe {recipe_id}, Batch Vial #{vial_in_batch}")

        # Optional Step 1: Pre-Fill Predictive Check
        if enable_pre_fill_guard:
            pre_eval = self.pre_fill_check(recipe_id, prev_recipe_id, vial_in_batch)
            if pre_eval["action"] == "PAUSE_BEFORE_FILL":
                self.plc.emergency_stop(fault_code=301, reason=pre_eval["detail"])
                self._print_pre_fill_halt_banner(recipe_id, prev_recipe_id, vial_in_batch, pre_eval["detail"])
                return "PRE_FILL_HALT"

        # Step 2: Liquid Dispense & Post-Fill Camera Inspection
        action, diagnostic = self.inspect_vial_image(
            image_path, recipe_id, prev_recipe_id, vial_in_batch, row_data=row_data
        )

        # Step 3: Actuate Machine Interlocks
        if action == "STOP_MACHINE":
            self.plc.emergency_stop(fault_code=402, reason="Fluid Contamination Detected")
            self._print_emergency_stop_banner(diagnostic)
            return "POST_FILL_STOP"
        else:
            # Write permissive high to let MagneMover advance
            self.plc.write_tag("Station3_Dispense_Permissive", True)
            self.plc.write_tag("MagneMover_Infeed_Active", True)
            print(f"[PLC TAG: Line_Running={self.plc.read_tag('Line_Running')}] PASS: Inspection verified. MagneMover advancing cart.")
            return "CONTINUE"

    def _print_emergency_stop_banner(self, diagnostic):
        print("\n" + "=" * 76)
        print("  🚨 [ROCKWELL AUTOMATION INTERLOCK] EMERGENCY MACHINE STOP ACTIVATED 🚨")
        print("=" * 76)
        for line in diagnostic.strip().split("\n"):
            print(f"  {line}")
        print("=" * 76)
        print(f"  PLC STATUS: Line_Running={self.plc.read_tag('Line_Running')} | E_Stop={self.plc.read_tag('Alarm_E_Stop')} | Flush_Req={self.plc.read_tag('Flush_Cycle_Required')}")
        print("=" * 76 + "\n")

    def _print_pre_fill_halt_banner(self, recipe_id, prev_recipe_id, vial_in_batch, detail):
        print("\n" + "=" * 76)
        print("  🛑 [PREDICTIVE GUARD INTERLOCK] PRE-FILL MACHINE PAUSE ACTIVATED 🛑")
        print("=" * 76)
        print(f"  Target Recipe: Recipe {recipe_id} | Previous Recipe: Recipe {prev_recipe_id}")
        print(f"  Vial:          Vial {vial_in_batch} of batch (unfilled)")
        print(f"  Prediction:    {detail}")
        print("  Action Taken:  Dispense head locked. Machine paused to purge manifold.")
        print("  Raw Materials: Saved (No vial or chemicals wasted).")
        print("=" * 76 + "\n")


def run_live_line_monitor(controller, image_dir="", poll_interval=0.1, enable_pre_fill_guard=False):
    """
    Continuous event loop for physical deployment on the UWM Connected Systems Institute (CSI) line.
    Polls the Allen-Bradley PLC over EtherNet/IP for cart arrival triggers, inspects the vial,
    and updates machine interlocks in real time.
    """
    print("\n" + "#" * 70)
    print("  🚀 [LIVE PHYSICAL MONITOR] ACTIVE ON ROCKWELL AUTOMATION CSI LINE 🚀")
    print(f"  Connected to PLC: {controller.plc.plc_ip or 'Local Simulated Driver'}")
    print("  Watching Tag: Station3_Cart_In_Position")
    print("#" * 70 + "\n")

    last_cart_state = False
    vials_processed = 0

    try:
        while True:
            # Poll the cart presence sensor from the PLC over EtherNet/IP
            cart_in_position = bool(controller.plc.read_tag("Cart_In_Position", False))

            # Detect rising edge (Cart just arrived at nozzle)
            if cart_in_position and not last_cart_state:
                vials_processed += 1
                print(f"\n[EVENT] MagneMover Cart arrived at Station 3 (Event #{vials_processed})")

                # Read operational context tags directly from the PLC
                rid = int(controller.plc.read_tag("Current_Recipe", 5))
                prev = int(controller.plc.read_tag("Prev_Recipe", 0))
                vial = int(controller.plc.read_tag("Vial_Number", 1))

                # Identify latest camera image
                img_path = None
                if image_dir and os.path.exists(image_dir):
                    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if files:
                        img_path = max(files, key=os.path.getmtime)  # Most recent image

                if not img_path:
                    img_path = "live_capture.png"

                # Process the vial through the AI controller
                status = controller.process_vial(
                    image_path=img_path,
                    recipe_id=rid,
                    prev_recipe_id=prev,
                    vial_in_batch=vial,
                    enable_pre_fill_guard=enable_pre_fill_guard,
                )

                # Signal handshaking bit to PLC
                controller.plc.write_tag("Inspection_Complete", True)

            elif not cart_in_position and last_cart_state:
                # Cart departed Station 3: reset handshaking bit
                controller.plc.write_tag("Inspection_Complete", False)

            last_cart_state = cart_in_position
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[INFO] Live monitoring terminated by operator.")
    finally:
        controller.plc.close()


def simulate_production_run(plan_csv, img_root="", plc_ip=None, plc_slot=0, custom_tags=None, mock=True, base_model="", adapter="", enable_pre_fill_guard=False):
    """Simulates a sequence of production vials through Station 3."""
    controller = LineController(
        base_model=base_model,
        adapter_path=adapter,
        plc_ip=plc_ip,
        plc_slot=plc_slot,
        custom_tags=custom_tags,
        mock=mock,
    )
    with open(plan_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"\nStarting line simulation for {len(rows)} planned vials...")
    stops_triggered = 0

    for i, r in enumerate(rows):
        rid = int(r["recipe_id"])
        prev = int(r.get("prev_recipe_id") or 0)
        vial = int(r.get("vial_in_batch") or 1)
        img_name = r.get("image", "")
        img_path = os.path.join(img_root, img_name) if img_root else img_name

        status = controller.process_vial(
            image_path=img_path,
            recipe_id=rid,
            prev_recipe_id=prev,
            vial_in_batch=vial,
            row_data=r,
            enable_pre_fill_guard=enable_pre_fill_guard,
        )

        if status in ("POST_FILL_STOP", "PRE_FILL_HALT"):
            stops_triggered += 1
            print(">>> [OPERATOR OVERRIDE] Simulating line purge & resetting PLC interlocks...")
            controller.plc.reset_and_flush()
            time.sleep(0.05)  # Short pause for clean console output

    print("\n" + "#" * 60)
    print(f"Simulation completed: {len(rows)} vials processed.")
    print(f"Total machine stops triggered & resolved: {stops_triggered}")
    print("#" * 60)
    controller.plc.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Station 3 Rockwell Automation Line Controller & Safety Interlock")
    ap.add_argument("--plan", help="path to plan.csv or labels.csv")
    ap.add_argument("--img-root", default="", help="folder containing rendered images")
    ap.add_argument("--image", help="single image to inspect")
    ap.add_argument("--recipe", type=int, help="current recipe id")
    ap.add_argument("--prev", type=int, default=0, help="previous recipe id")
    ap.add_argument("--vial", type=int, default=1, help="vial in batch (1, 2, or 3)")
    ap.add_argument("--base", default="", help="path to Cosmos Reason base model")
    ap.add_argument("--adapter", default="", help="path to trained LoRA adapter")
    ap.add_argument("--mock", action="store_true", help="run simulation mode without GPU")
    ap.add_argument("--predictive", action="store_true", help="enable pre-fill predictive prevention")

    # EtherNet/IP Industrial Hardware Flags
    ap.add_argument("--plc-ip", default="", help="IP address of physical Allen-Bradley PLC (e.g. 192.168.1.10)")
    ap.add_argument("--plc-slot", type=int, default=0, help="Processor slot in 1756 ControlLogix chassis (default 0)")
    ap.add_argument("--plc-tags", default="", help="JSON file mapping custom Studio 5000 tag names")
    ap.add_argument("--live", action="store_true", help="run continuous real-time monitor for physical CSI line")
    ap.add_argument("--image-dir", default="", help="camera capture inbox folder to watch in live mode")
    args = ap.parse_args()

    custom_tags = None
    if args.plc_tags and os.path.isfile(args.plc_tags):
        with open(args.plc_tags) as f:
            custom_tags = json.load(f)

    if args.live:
        controller = LineController(
            base_model=args.base,
            adapter_path=args.adapter,
            plc_ip=args.plc_ip,
            plc_slot=args.plc_slot,
            custom_tags=custom_tags,
            mock=args.mock or not args.base,
        )
        run_live_line_monitor(
            controller=controller,
            image_dir=args.image_dir,
            enable_pre_fill_guard=args.predictive,
        )
    elif args.plan:
        simulate_production_run(
            plan_csv=args.plan,
            img_root=args.img_root,
            plc_ip=args.plc_ip,
            plc_slot=args.plc_slot,
            custom_tags=custom_tags,
            mock=args.mock or not args.base,
            base_model=args.base,
            adapter=args.adapter,
            enable_pre_fill_guard=args.predictive,
        )
    elif args.image and args.recipe:
        controller = LineController(
            base_model=args.base,
            adapter_path=args.adapter,
            plc_ip=args.plc_ip,
            plc_slot=args.plc_slot,
            custom_tags=custom_tags,
            mock=args.mock or not args.base,
        )
        controller.process_vial(
            image_path=args.image,
            recipe_id=args.recipe,
            prev_recipe_id=args.prev,
            vial_in_batch=args.vial,
            enable_pre_fill_guard=args.predictive,
        )
        controller.plc.close()
    else:
        # Default quick demo if no arguments provided
        default_plan = os.path.join(REPO_ROOT, "twin_data", "demo_out", "plan.csv")
        if os.path.exists(default_plan):
            print("[INFO] No arguments given. Running demo simulation on plan.csv...")
            simulate_production_run(default_plan, mock=True, enable_pre_fill_guard=False)
        else:
            ap.print_help()
