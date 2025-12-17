import asyncio
import websockets
import json
import subprocess
import base64
import sys
import os
import time
import hashlib
import uiautomator2 as u2
from uiautomator2 import Device
import cv2
import numpy as np

# ==============================================================================
# 配置区域
# ==============================================================================
SERVER_URL = "ws://localhost:8000/ws" 
DEVICE_ID = "" # 自动选择

# 🚀 Resolve ADB Path
ADB_PATH = "adb"
if os.path.exists(os.path.join(os.getcwd(), "platform-tools", "adb")):
    ADB_PATH = os.path.join(os.getcwd(), "platform-tools", "adb")

import uuid

class EdgeClient:
    def __init__(self, client_id: str, target_serial: str = None):
        self.client_id = client_id
        self.target_serial = target_serial
        self.mock_mode = False
        self.device: Device = None
        self.smart_wait_enabled = False # 🚀 Disabled by default for speed
        self._init_u2()
        print(f"🆔 Client ID initialized as: {self.client_id}")

    def _init_u2(self):
        try:
            # 🆕 Restart ADB to ensure fresh connection (Only if not targeting specific device to avoid race)
            if not self.target_serial:
                print("🔄 Restarting ADB...")
                subprocess.run([ADB_PATH, "kill-server"])
                subprocess.run([ADB_PATH, "start-server"])
            
            # 🆕 Get device list and pick first one to avoid "more than one device" error
            result = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:] # Skip header
            devices = [line.split()[0] for line in lines if line.strip() and "device" in line]
            
            target_device = None
            
            if self.target_serial:
                if self.target_serial in devices:
                    target_device = self.target_serial
                    print(f"📱 Target Device Found: {target_device}")
                else:
                    print(f"❌ Target Device {self.target_serial} NOT FOUND! Available: {devices}")
                    # 🆕 Exit if specific target is missing (User Requirement: No Mock for missing device)
                    print("⚠️ Target device missing. Exiting...")
                    sys.exit(1)
            
            if not target_device and not self.mock_mode:
                if devices:
                    # 🆕 Prioritize Physical Devices (exclude emulator)
                    physical_devices = [d for d in devices if not d.startswith("emulator")]
                    if physical_devices:
                        target_device = physical_devices[0]
                        print(f"📱 Found devices: {devices}, preferring physical: {target_device}")
                    else:
                        # 🚫 Refuse Emulator
                        print(f"⚠️ Only emulators found: {devices}. User requested PHYSICAL DEVICE ONLY.")
                        print("❌ Please connect a real Android device via USB.")
                        sys.exit(1)
                else:
                    print("⚠️ No devices found. Exiting...")
                    sys.exit(1)

            # Connect to device via USB
            if target_device:
                self.device_serial = target_device
                self.device = u2.connect(target_device)
            else:
                self.device_serial = None
                self.device = u2.connect() # Fallback to default behavior
                
            print(f"✅ Uiautomator2 Connected: {self.device.info}")
            
            # 🆕 Enable FastInput IME to hide keyboard and support fast typing
            try:
                print("⌨️ Enabling FastInput IME (Hidden Keyboard)...")
                self.device.set_fastinput_ime(True)
            except Exception as e:
                print(f"⚠️ Failed to set FastInput IME: {e}")

            # Optional: Disable uiautomator2's default watcher to avoid interference, 
            # but keep it if we want it to handle system popups.
            # self.device.watcher.stop()
        except Exception as e:
            print(f"⚠️ Uiautomator2 Connection Failed: {e}")
            self.mock_mode = True

    async def _run_adb_async(self, args, timeout=5):
        """Helper to run ADB commands asynchronously without blocking the event loop"""
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                return stdout
            except asyncio.TimeoutError:
                print(f"⚠️ ADB Timeout: {args}")
                try:
                    process.kill()
                except:
                    pass
                return None
        except Exception as e:
            print(f"❌ Async ADB Error: {e}")
            return None

    def wait_for_ui_stability(self, timeout=3.0):
        """Visual Smart Wait: Wait until screen content (XML) stops changing"""
        # 🆕 PERFORMANCE FIX: Smart Wait is disabled by default to speed up "Fast Traversal"
        # Only enable it if explicitly requested or for "tap" actions where stability matters.
        if self.mock_mode or not self.smart_wait_enabled:
            # Simple small sleep is enough for most actions
            time.sleep(0.1) # 🚀 Reduced from 0.5s to 0.1s for speed
            return

        print("👀 Smart Wait: Checking for UI stability...")
        start_time = time.time()
        last_hash = ""
        stable_frames = 0
        
        while time.time() - start_time < timeout:
            try:
                # Use XML dump as stability proxy (reliable but potentially slow)
                # If dump fails or takes too long, we might timeout, which is fine.
                xml = self.device.dump_hierarchy()
                current_hash = hashlib.md5(xml.encode()).hexdigest()
                
                if current_hash == last_hash:
                    stable_frames += 1
                    # Require 2 consecutive matching frames to be sure
                    if stable_frames >= 1: 
                        elapsed = time.time() - start_time
                        print(f"✅ UI Stable in {elapsed:.2f}s")
                        return
                else:
                    stable_frames = 0
                    last_hash = current_hash
                
                # Small sleep to avoid hammering
                time.sleep(0.3)
            except Exception as e:
                print(f"⚠️ Smart Wait Error: {e}")
                break


    async def capture_screenshot(self) -> str:
        """截取屏幕并转为 Base64 (High Quality PNG for AI)"""
        # print("📸 Capturing screenshot...")
        
        if self.mock_mode:
            # ... (Mock Logic)
            return ""

        try:
            # 🚀 OPTIMIZATION: Use ADB screencap instead of u2.screenshot
            # uiautomator2's screenshot is slow (2-3s) because it syncs with Accessibility service.
            # ADB raw screencap is much faster (0.5s - 1s).
            
            if hasattr(self, 'device_serial') and self.device_serial:
                # Reuse the raw ADB logic from capture_preview but keep high quality
                cmd = [ADB_PATH, "-s", self.device_serial, "shell", "screencap", "-p"]
                
                # 🆕 ASYNC ADB Execution
                png_data = await self._run_adb_async(cmd, timeout=10)
                
                if png_data:
                    # 🚀 COMPRESSION OPTIMIZATION: Convert PNG to JPEG (Quality 70)
                    # Raw PNG is too big (3-5MB), JSON dump takes too long.
                    # JPEG compression (0.1s) saves 1.5s in transmission/serialization.
                    
                    try:
                        from PIL import Image
                        import io
                        
                        # Run blocking image processing in a separate thread to avoid blocking loop
                        def process_image():
                            image = Image.open(io.BytesIO(png_data))
                            if image.mode in ('RGBA', 'P'):
                                image = image.convert('RGB')
                                
                            # Downscale for AI (Faster transmission, enough for vision)
                            # Standard Android is 1080x2400. Resize to 540x1200.
                            image.thumbnail((540, 1200))
                            
                            buffered = io.BytesIO()
                            image.save(buffered, format="JPEG", quality=70)
                            return base64.b64encode(buffered.getvalue()).decode('utf-8')

                        return await asyncio.to_thread(process_image)
                        
                    except Exception as e:
                        print(f"⚠️ Compression Error: {e}, falling back to raw PNG")
                        return base64.b64encode(png_data).decode('utf-8')
            
            # Fallback to U2
            image = await asyncio.to_thread(self.device.screenshot, format='pillow')
            import io
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"❌ Screenshot Error: {e}")
            return ""

    async def capture_preview(self) -> str:
        """截取屏幕预览 (Low Quality JPEG for Streaming)"""
        if self.mock_mode:
            # Use same generator as screenshot but maybe faster?
            # For simplicity, reuse screenshot logic but save as JPEG
            return await self.capture_screenshot()

        # print("DEBUG: Entering capture_preview", flush=True)
        try:
            # 🚀 Optimization: Use ADB directly (Robust & Fast)
            if hasattr(self, 'device_serial') and self.device_serial:
                cmd = [ADB_PATH, "-s", self.device_serial, "shell", "screencap", "-p"]
                
                # 🆕 ASYNC ADB Execution
                png_data = await self._run_adb_async(cmd, timeout=5)
                
                if not png_data:
                    # print("⚠️ ADB Screencap returned empty", flush=True)
                    return ""
                
                # Convert to JPEG (Offload to thread)
                try:
                    def process_preview():
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(png_data))
                        if image.mode in ('RGBA', 'P'):
                            image = image.convert('RGB')
                            
                        image.thumbnail((360, 800))
                        
                        buffered = io.BytesIO()
                        image.save(buffered, format="JPEG", quality=50)
                        return base64.b64encode(buffered.getvalue()).decode('utf-8')

                    return await asyncio.to_thread(process_preview)
                    
                except Exception as e:
                    print(f"⚠️ Image Conversion Error: {e}", flush=True)
                    return ""

            # Fallback to U2 if ADB serial not found
            # print("📸 Capture: Using U2 fallback...", flush=True)
            image = await asyncio.to_thread(self.device.screenshot, format='pillow')
            
            def process_u2_preview(img):
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.thumbnail((360, 800))
                import io
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=50)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
                
            return await asyncio.to_thread(process_u2_preview, image)
            
        except Exception as e:
            print(f"❌ Preview Error: {e}", flush=True)
            return ""

    def get_ui_hierarchy(self) -> str:
        """获取 UI 结构 XML"""
        if self.mock_mode:
            # Return a fake XML that matches the drawn rectangles
            xml = '<hierarchy rotation="0"><node index="0" text="" resource-id="" class="android.widget.FrameLayout" package="com.mock" content-desc="" checkable="false" checked="false" clickable="false" enabled="true" focusable="false" focused="false" scrollable="false" long-clickable="false" password="false" selected="false" bounds="[0,0][720,1280]">'
            for i in range(5):
                y = 300 + i * 150
                xml += f'<node index="{i+1}" text="Mock Item {i+1}" resource-id="com.mock:id/btn_{i}" class="android.widget.Button" package="com.mock" content-desc="" checkable="false" checked="false" clickable="true" enabled="true" focusable="true" focused="false" scrollable="false" long-clickable="false" password="false" selected="false" bounds="[100,{y}][620,{y+100}]" />'
            xml += '</node></hierarchy>'
            return xml
            
        try:
            # Uiautomator2 dump
            return self.device.dump_hierarchy()
        except Exception as e:
            print(f"❌ XML Dump Error: {e}")
            return ""

    async def get_current_app(self):
        """获取当前运行的 App 包名和 Activity"""
        if self.mock_mode:
            return {}
        try:
            # 🚀 ULTRA-OPTIMIZATION: Use ADB shell dumpsys activity top (Faster & Simpler)
            
            if not getattr(self, 'device_serial', None):
                 return await asyncio.to_thread(self.device.app_current)

            # Try a very lightweight command first:
            # adb shell "dumpsys activity recents" (Very fast, ~0.07s)
            
            cmd = [ADB_PATH, "-s", self.device_serial, "shell", "dumpsys activity recents"]
            
            # 🆕 ASYNC ADB Execution
            output = await self._run_adb_async(cmd, timeout=2)
            
            if output:
                lines = output.decode('utf-8').split('\n')
                # Parse: "    topActivity={com.package/com.package.Activity}" inside "RecentTaskInfo #0" block
                # Simplified: Look for first occurrence of "topActivity={"
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("topActivity={"):
                        # Format: topActivity={com.example/com.example.MainActivity}
                        content = line.split("={")[1].split("}")[0]
                        parts = content.split("/")
                        if len(parts) == 2:
                            pkg = parts[0]
                            act = parts[1]
                            if act.startswith("."):
                                act = pkg + act
                            return {"package": pkg, "activity": act}
                        break
            
            # Fallback to u2 if fast method fails
            # print("⚠️ Fast App Info failed, using U2 fallback (Slow)...")
            return await asyncio.to_thread(self.device.app_current)
            
        except Exception as e:
            return {}

    def find_image_on_screen(self, template_b64: str, threshold: float = 0.8) -> tuple:
        """
        Visual Anchor: Find a template image on the current screen using OpenCV.
        Returns (x, y) center coordinates or None if not found.
        """
        try:
            # 1. Capture current screen (Raw PNG for accuracy)
            # Use sync screenshot for this blocking operation (or we should await it if called from async)
            # But execute_action is sync, so we use sync calls.
            # Using ADB screencap is faster.
            cmd = [ADB_PATH, "-s", self.device_serial, "shell", "screencap", "-p"]
            result = subprocess.run(cmd, capture_output=True)
            screen_data = result.stdout
            
            if not screen_data:
                print("⚠️ Screencap failed for template matching")
                return None

            # 2. Decode Screen
            nparr_screen = np.frombuffer(screen_data, np.uint8)
            screen_img = cv2.imdecode(nparr_screen, cv2.IMREAD_COLOR) # BGR
            
            # 3. Decode Template
            if "," in template_b64:
                template_b64 = template_b64.split(",")[1]
            template_data = base64.b64decode(template_b64)
            nparr_template = np.frombuffer(template_data, np.uint8)
            template_img = cv2.imdecode(nparr_template, cv2.IMREAD_COLOR)
            
            # 4. Template Matching
            # Ensure template is not larger than screen
            if template_img.shape[0] > screen_img.shape[0] or template_img.shape[1] > screen_img.shape[1]:
                print("⚠️ Template matches failed: Template larger than screen")
                return None
                
            result = cv2.matchTemplate(screen_img, template_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            print(f"👁️ Template Match Score: {max_val:.4f} (Threshold: {threshold})")
            
            if max_val >= threshold:
                # Calculate Center
                h, w = template_img.shape[:2]
                top_left = max_loc
                center_x = top_left[0] + w // 2
                center_y = top_left[1] + h // 2
                return (center_x, center_y)
            else:
                return None
                
        except Exception as e:
            print(f"❌ Template Matching Error: {e}")
            return None

    def execute_action(self, action: dict):
        """执行 AI 下发的指令"""
        act_type = action.get("action")
        reason = action.get("reason", "")
        print(f"🤖 Executing: {act_type} - {reason}")
        
        if self.mock_mode:
            return

        try:
            if act_type == "tap":
                self.device.click(action['x'], action['y'])
            elif act_type == "long_press":
                self.device.long_click(action['x'], action['y'], 1.0)
            elif act_type == "input":
                # u2 supports fast input
                self.device.send_keys(action['text'], clear=True)
            elif act_type == "swipe":
                self.device.swipe(action['x1'], action['y1'], action['x2'], action['y2'])
            elif act_type == "scroll":
                # Use scroll forward (down)
                # self.device(scrollable=True).scroll.forward()
                # Or manual swipe for better control
                w, h = self.device.window_size()
                self.device.swipe(w//2, h*0.8, w//2, h*0.2)
            elif act_type == "key_event":
                 # Use shell command for direct OS-level key event injection (More robust)
                 keycode = action['keycode']
                 print(f"⌨️ Injecting Key Event: {keycode}")
                 self.device.shell(f"input keyevent {keycode}")
            elif act_type == "shell":
                 # 🆕 Execute arbitrary shell command (e.g. am start)
                 cmd = action['command']
                 print(f"🐚 Executing Shell Command: {cmd}")
                 self.device.shell(cmd)
            elif act_type == "app_start":
                 # Launch app
                 self.device.app_start(action['package'])
            elif act_type == "template_tap":
                 # 🆕 Visual Anchor Tap
                 template = action.get('template') # Base64
                 threshold = action.get('threshold', 0.8)
                 
                 print(f"👁️ Visual Search: Looking for template (len={len(template)})...")
                 coords = self.find_image_on_screen(template, threshold)
                 
                 if coords:
                     x, y = coords
                     print(f"🎯 Visual Target Found at ({x}, {y}). Tapping...")
                     self.device.click(x, y)
                 else:
                     print("❌ Visual Target NOT Found.")
                     # Fallback? Maybe raise error or let server handle
                     
        except Exception as e:
            print(f"❌ Action Execution Error: {e}")

        # 🆕 Smart Wait after action
        self.wait_for_ui_stability()

async def start_client():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, help="Specific ADB Device Serial to connect", default=None)
    args = parser.parse_args()

    import socket
    hostname = socket.gethostname()
    # 🚀 Use UUID to ensure unique connection ID even if process restarts
    import uuid
    
    # 🆕 Append Serial to Client ID for easy identification
    suffix = f"_{args.serial}" if args.serial else f"_{uuid.uuid4().hex[:4]}"
    client_id = f"client_{hostname}{suffix}"
    
    uri = f"{SERVER_URL}/{client_id}"
    print(f"🔌 Connecting to Cloud Brain: {uri} ...")
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Connected! Waiting for commands...")
                client = EdgeClient(client_id, target_serial=args.serial)
                
                # 🚀 启动后台流式传输任务 (Streaming Task)
                async def stream_screen():
                    print("🎥 Streaming started...")
                    frame_count = 0
                    while True:
                        try:
                            # Use optimized preview capture
                            b64_img = await client.capture_preview()
                            if b64_img:
                                # print(f"DEBUG: Sending frame len={len(b64_img)}", flush=True)
                                resp = {"type": "stream_frame", "data": b64_img}
                                await websocket.send(json.dumps(resp))
                                frame_count += 1
                                if frame_count % 50 == 0:
                                    print(f"🎥 Sent {frame_count} frames...", flush=True)
                            else:
                                # print("⚠️ Preview capture returned empty", flush=True)
                                pass
                            
                            await asyncio.sleep(0.2) 
                        except Exception as e:
                            print(f"❌ Streaming Error: {e}")
                            break
                    print("🎥 Streaming stopped")
                
                # 启动流
                stream_task = asyncio.create_task(stream_screen())
                
                # 🆕 Heartbeat Task
                async def send_heartbeat():
                    while True:
                        try:
                            await websocket.send(json.dumps({"type": "heartbeat"}))
                            await asyncio.sleep(5)
                        except Exception:
                            break
                
                heartbeat_task = asyncio.create_task(send_heartbeat())
                
                while True:
                    try:
                        # 等待 Server 指令
                        message = await websocket.recv()
                        cmd = json.loads(message)
                        
                        if cmd["type"] == "capture_screenshot":
                            # 收到截图指令
                            t0 = time.time()
                            b64_img = await client.capture_screenshot()
                            t1 = time.time()
                            # Run XML dump in thread to avoid blocking if we have background tasks
                            xml_data = await asyncio.to_thread(client.get_ui_hierarchy)
                            t2 = time.time()
                            # 🆕 Get App Info
                            app_info = await client.get_current_app()
                            t3 = time.time()
                            
                            resp = {
                                "type": "screenshot_response", 
                                "id": cmd.get("id"),
                                "data": b64_img, 
                                "xml": xml_data, 
                                "app_info": app_info # Send package/activity info
                            }
                            
                            t4 = time.time()
                            json_str = json.dumps(resp)
                            t5 = time.time()
                            
                            print(f"⏱️ PERF DEBUG: Cap={t1-t0:.2f}s, XML={t2-t1:.2f}s, Info={t3-t2:.2f}s, JSON={t5-t4:.2f}s, Total={t5-t0:.2f}s", flush=True)
                            
                            await websocket.send(json_str)
                            
                        elif cmd["type"] == "execute_action":
                            # 收到动作指令
                            client.execute_action(cmd["payload"])
                            # 回传执行完成
                            await websocket.send(json.dumps({"type": "action_done"}))
                        
                        elif cmd["type"] == "reconnect_device":
                            print("🔄 Received Reconnect Command from Server...")
                            client._init_u2()
                            await websocket.send(json.dumps({"type": "log", "message": "Device Reconnection Attempted"}))
                            
                        elif cmd["type"] == "update_config":
                            cfg = cmd["payload"]
                            print(f"⚙️ Received Config Update: {cfg}")
                            if "smart_wait" in cfg:
                                client.smart_wait_enabled = cfg["smart_wait"]
                                print(f"✅ Smart Wait set to: {client.smart_wait_enabled}")
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("❌ Connection closed by server")
                        break
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        break
                
                heartbeat_task.cancel()
                stream_task.cancel()

        except Exception as e:
            print(f"⚠️ Connection failed: {e}. Retrying in 3s...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(start_client())
    except KeyboardInterrupt:
        print("\nBye!")
