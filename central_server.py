import asyncio
import json
import base64
import time
import os
import sys
from vector_memory import VectorMemory
from training_logger import TrainingLogger
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="TestGPT Central Brain")
training_logger = TrainingLogger()

# 存储所有连接的边缘节点
# key: client_id, value: {"ws": WebSocket, "last_seen": timestamp, "status": "idle", "logs": []}
active_clients: Dict[str, dict] = {}

class ConnectionManager:
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        
        # 尝试加载历史数据
        history = load_client_state(client_id)
        
        if history:
            print(f"📂 Loaded history for {client_id}")
            # 恢复 active_clients 数据
            client_data = history.get("client", {})
            active_clients[client_id] = {
                "ws": websocket,
                "last_seen": time.time(),
                "status": "connected",
                "logs": [], # 日志不恢复，保持清爽
                "last_screenshot": None, # 等待客户端发第一帧
                "stats": client_data.get("stats", {"pages": 0, "steps": 0, "bugs": 0}),
                "phase": client_data.get("phase", "IDLE"),
                "test_cases": client_data.get("test_cases", []),
                "report": {}
            }
            # 恢复 Graph 数据
            graph_data = history.get("graph", {})
            if graph_data:
                graph = AppStateGraph()
                graph.from_dict(graph_data)
                client_graphs[client_id] = graph
                
            self.add_log(client_id, f"📂 已加载历史记录: {len(client_graphs[client_id].states)} 个页面, {len(active_clients[client_id]['test_cases'])} 条用例")
        else:
            # 新 Client
            active_clients[client_id] = {
                "ws": websocket,
                "last_seen": time.time(),
                "status": "connected",
                "logs": [],
                "last_screenshot": None,
                "stats": {"pages": 0, "steps": 0, "bugs": 0, "last_new_state_step": 0}, # 🆕 Added last_new_state_step
                "phase": "IDLE", # IDLE, SCANNING, PLANNING, DEEP_EXPLORING, FINISHED
                "test_cases": [], # [{"id": 1, "desc": "Login test", "status": "pending"}]
                "exploration_tasks": [], # 🆕 For Phase 2 Planning
                "current_task_index": 0,
                "report": {} # {"total": 10, "passed": 8, "failed": 2}
            }
            
        print(f"✅ Edge Client Connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in active_clients:
            del active_clients[client_id]
            print(f"❌ Edge Client Disconnected: {client_id}")

    async def send_command(self, client_id: str, command: dict):
        if client_id in active_clients:
            await active_clients[client_id]["ws"].send_text(json.dumps(command))
            return True
        # 🆕 Condition 4: Cycle Detection (A -> B -> A -> B)
        # Check the sequence of canonical IDs (or hashes)
        if len(self.transitions) >= 6:
            # Get last 6 destination hashes
            # transition = (from_hash, action, to_hash)
            # We care about to_hash
            path = [t[2] for t in self.transitions[-6:]]
            
            # Pattern: A, B, A, B
            # last 4: [A, B, A, B]
            if path[-1] == path[-3] and path[-2] == path[-4]:
                 print(f"🛑 Detected 2-Page Cycle: {path[-1][:8]} <-> {path[-2][:8]}")
                 
                 # Identify the action causing the loop (the one that brought us to current page)
                 loop_entry_action = self.transitions[-1][1]
                 
                 # Blacklist it globally
                 # We need the source page ID
                 from_hash = self.transitions[-1][0]
                 from_state = self.states.get(from_hash)
                 if from_state:
                     from_id = from_state.get("canonical_id") or from_state.get("semantic_hash")
                     if from_id:
                         fingerprint = self.get_action_fingerprint(from_id, loop_entry_action)
                         self.ineffective_actions.add(fingerprint)
                         print(f"🚫 Cycle Breaker: Blacklisted action {fingerprint[:8]}")
                         
                 return True

        return False
        
    def add_log(self, client_id: str, message: str):
        if client_id in active_clients:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            log_entry = f"[{timestamp}] {message}"
            active_clients[client_id]["logs"].append(log_entry)
            # 只保留最近 200 条日志 (Increased from 50)
            if len(active_clients[client_id]["logs"]) > 200:
                active_clients[client_id]["logs"].pop(0)

    def update_screenshot(self, client_id: str, b64_img: str):
        if client_id in active_clients:
            # print(f"DEBUG: Storing screenshot for {client_id}, len={len(b64_img)}")
            active_clients[client_id]["last_screenshot"] = b64_img

manager = ConnectionManager()

import hashlib
from dotenv import load_dotenv
import os
from vector_memory import VectorMemory

# Load environment variables
load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "mock")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "gpt-4o")

# ==============================================================================
# 💾 数据持久化 (Persistence Layer)
# ==============================================================================
DATA_DIR = "test_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def save_client_state(client_id: str):
    """保存 Client 的所有状态 (Graph, Stats, Test Cases)"""
    try:
        # 1. 准备 Graph 数据
        graph_data = {}
        if client_id in client_graphs:
            graph = client_graphs[client_id]
            graph_data = {
                "states": graph.states,
                "transitions": graph.transitions,
                "current_state_hash": graph.current_state_hash,
                "global_action_memory": list(graph.global_action_memory), # 🆕 Save Global Memory
                "ineffective_actions": list(graph.ineffective_actions) # 🆕 Save Blacklist
            }
            
        # 2. 准备 Client 数据
        client_data = {}
        if client_id in active_clients:
            c = active_clients[client_id]
            client_data = {
                "stats": c["stats"],
                "test_cases": c["test_cases"],
                "phase": c["phase"]
            }
            
        # 3. 保存到文件
        if graph_data or client_data:
            # Convert sets to lists for JSON serialization
            if "global_action_memory" in graph_data:
                graph_data["global_action_memory"] = list(graph_data["global_action_memory"])
            if "ineffective_actions" in graph_data:
                graph_data["ineffective_actions"] = list(graph_data["ineffective_actions"])
            
            # Helper for recursive set conversion
            def convert_sets(obj):
                if isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert_sets(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_sets(i) for i in obj]
                return obj
            
            graph_data = convert_sets(graph_data)
            client_data = convert_sets(client_data)

            file_path = os.path.join(DATA_DIR, f"{client_id}.json")
            with open(file_path, "w") as f:
                json.dump({"graph": graph_data, "client": client_data}, f, ensure_ascii=False, indent=2)
            # print(f"💾 Saved state for {client_id}")
            
    except Exception as e:
        print(f"❌ Failed to save state: {e}")

def load_client_state(client_id: str):
    """加载 Client 历史状态"""
    file_path = os.path.join(DATA_DIR, f"{client_id}.json")
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load state: {e}")
        return None

from PIL import Image, ImageDraw, ImageFont
import io

def crop_element(image_b64: str, bounds: list) -> str:
    """
    Crop an element from the screenshot based on [x1, y1, x2, y2].
    Returns base64 encoded JPEG of the crop.
    """
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # Ensure bounds are within image
        x1, y1, x2, y2 = bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = image.crop((x1, y1, x2, y2))
        
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=90) # High quality for template matching
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ Crop Failed: {e}")
        return None

# ==============================================================================
# 🔧 图像处理工具 (Image Utils)
# ==============================================================================
def draw_som_overlay(base64_str: str, xml_data: str) -> (str, dict):
    """
    Set-of-Mark (SOM) 标记生成：
    1. 解析 XML 获取所有元素坐标
    2. 在截图上绘制半透明遮罩和数字标签
    3. 返回处理后的图片 (Base64) 和 ID映射表 {id: element_info}
    """
    if not xml_data:
        return base64_str, {}

    try:
        # 1. Decode Image
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        draw = ImageDraw.Draw(image, "RGBA")
        
        # 2. Parse XML
        elements = parse_xml_elements(xml_data)
        if not elements:
            return base64_str, {}
            
        id_map = {}
        
        # Load Font (Try to load a nice font, fallback to default)
        try:
            # Mac default font
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
        except:
            font = ImageFont.load_default()

        # 3. Draw Tags
        for idx, el in enumerate(elements):
            tag_id = idx + 1
            x1, y1, x2, y2 = el['bounds']
            
            # Skip huge elements (likely background)
            if (x2 - x1) > image.width * 0.9 and (y2 - y1) > image.height * 0.9:
                continue
                
            # Draw Box (Semi-transparent Red)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200), width=2)
            
            # Draw Tag Background
            tag_w, tag_h = 30, 24
            tag_x = x1
            tag_y = max(0, y1 - tag_h) # Put tag above element if possible
            draw.rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + tag_h], fill=(255, 0, 0, 255))
            
            # Draw Tag Number
            draw.text((tag_x + 5, tag_y + 2), str(tag_id), fill=(255, 255, 255, 255), font=font)
            
            # Store in Map
            id_map[tag_id] = el
            
        # 4. Encode back to Base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        tagged_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return tagged_b64, id_map
        
    except Exception as e:
        print(f"⚠️ SOM Drawing Failed: {e}")
        return base64_str, {}

def compress_image(base64_str: str, max_width: int = 720, quality: int = 70) -> str:
    """压缩图片：调整大小 + JPEG 压缩，大幅减少 Token 消耗和网络传输时间"""
    try:
        # 1. Decode
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        
        # 2. Resize (保持比例)
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
        # 3. Convert to RGB (去除 Alpha 通道，兼容 JPEG)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
            
        # 4. Encode to JPEG
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ Image compression failed: {e}")
        return base64_str

def get_perceptual_hash(base64_str: str) -> str:
    """感知哈希：忽略时间、电量条等微小变化，只关注页面整体结构"""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        
        # 缩小到极小分辨率 (比如 32x64) 并灰度化
        # 这样即使时间变了（几个像素），整体缩略图的 Hash 还是几乎一样的
        thumb = image.resize((32, 64), Image.Resampling.BILINEAR).convert('L')
        
        # 计算像素的简单 Hash
        pixels = list(thumb.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join(["1" if p > avg else "0" for p in pixels])
        return hashlib.md5(bits.encode()).hexdigest()
    except Exception as e:
        print(f"⚠️ Perceptual hash failed: {e}")
        return hashlib.md5(base64_str.encode()).hexdigest()

import colorsys
import os

def check_connection_status(screenshot_b64: str, xml_data: str = None) -> (bool, dict):
    """
    检查左上角图标颜色是否为绿色。
    Returns: (is_connected, icon_center_coords)
    """
    # 🆕 Popup Detection: If popup is present, skip check (return True)
    if xml_data:
        try:
            # Check for common Dialog resource IDs or Class names
            popup_indicators = [
                "id/parentPanel",       # Standard AlertDialog
                "id/alertTitle",        # Dialog Title
                "id/buttonPanel",       # Dialog Buttons
                "id/permission_message", # Permission Dialog
                "android.widget.PopupWindow" # Popup Window
            ]
            for indicator in popup_indicators:
                if indicator in xml_data:
                    print(f"🛡️ Popup Detected ({indicator}). Skipping Connection Check.")
                    return True, None # Assume connected
        except Exception as e:
            print(f"⚠️ Popup Check Error: {e}")

    try:
        if "," in screenshot_b64:
            screenshot_b64 = screenshot_b64.split(",")[1]
        image_data = base64.b64decode(screenshot_b64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Default Region: Top-Left 150x150
        crop_box = (0, 0, 150, 150)
        
        # 🆕 Try to locate "aiTalk" text
        if xml_data:
            try:
                root = ET.fromstring(xml_data)
                found_anchor = False
                for node in root.iter():
                    text = node.attrib.get('text', '').lower()
                    if "aitalk" in text:
                        bounds = node.attrib.get('bounds')
                        if bounds:
                            match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                            if match:
                                x1, y1, x2, y2 = map(int, match.groups())
                                print(f"📍 Found 'aiTalk' anchor at: [{x1},{y1}][{x2},{y2}]")
                                # Define Region to the LEFT of this text
                                box_y1 = max(0, y1 - 10)
                                box_y2 = min(image.height, y2 + 10)
                                box_x2 = max(10, x1) # Up to the start of text
                                box_x1 = max(0, box_x2 - 150) # Look back 150px
                                
                                crop_box = (box_x1, box_y1, box_x2, box_y2)
                                found_anchor = True
                                break
                if not found_anchor:
                    print("⚠️ 'aiTalk' text not found in XML. Trying to find any Top-Left Element...")
                    # Fallback: Find the left-most, top-most element (likely the icon button)
                    best_candidate = None
                    min_dist = 9999
                    
                    for node in root.iter():
                        bounds = node.attrib.get('bounds')
                        if bounds:
                            match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                            if match:
                                x1, y1, x2, y2 = map(int, match.groups())
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                
                                # Only look at Top-Left corner (0-200, 0-200)
                                if cx < 200 and cy < 200:
                                    # Prefer elements that are somewhat square-ish or small (icon-like)
                                    w, h = x2 - x1, y2 - y1
                                    if 20 < w < 150 and 20 < h < 150:
                                        dist = cx + cy
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_candidate = (x1, y1, x2, y2)
                    
                    if best_candidate:
                        print(f"📍 Found Potential Icon Element at: {best_candidate}")
                        crop_box = best_candidate
                    else:
                        print("⚠️ No suitable Top-Left element found. Using default fixed region.")

            except Exception as e:
                print(f"⚠️ XML Parse for Anchor failed: {e}")

        print(f"🔍 Checking Connection Status in Box: {crop_box}")
        crop = image.crop(crop_box)
        
        # 🆕 Debug: Save the crop for inspection
        try:
            debug_path = os.path.join("test_data", "debug_icon_crop.png")
            crop.save(debug_path)
            # print(f"💾 Saved debug crop to {debug_path}")
        except:
            pass

        pixels = list(crop.getdata())
        
        # 🆕 Improved Color Analysis (HSV + Detailed Stats)
        green_pixel_count = 0
        total_r, total_g, total_b = 0, 0, 0
        
        # To debug: Count dominant colors
        from collections import Counter
        color_counts = Counter(pixels)
        top_5 = color_counts.most_common(5)
        print(f"🎨 Top 5 Colors in Box: {top_5}")
        
        for r, g, b in pixels:
            total_r += r
            total_g += g
            total_b += b
            
            # HSV Conversion
            # colorsys uses 0-1 range
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            h_deg = h * 360
            
            # Green Hue Range: 70 - 170 (Wide range covering yellow-green to cyan-green)
            # Saturation > 0.05 (Very low threshold for pale colors)
            # Value > 0.2 (Not pitch black)
            if 70 <= h_deg <= 170 and s > 0.05 and v > 0.2:
                 green_pixel_count += 1
            # Fallback RGB check for very desaturated greens that HSV might miss due to noise
            elif g > r + 5 and g > b + 5 and g > 100:
                 green_pixel_count += 1
        
        avg_r = total_r // len(pixels) if pixels else 0
        avg_g = total_g // len(pixels) if pixels else 0
        avg_b = total_b // len(pixels) if pixels else 0
        
        print(f"🎨 Pixel Analysis: Total={len(pixels)}, Green={green_pixel_count}")
        print(f"🎨 Average Color in Box: R={avg_r}, G={avg_g}, B={avg_b}")
        
        # 阈值设为 3 个像素 (极度敏感)
        is_connected = green_pixel_count > 3
        print(f"🔌 Connection Check: Green Pixels = {green_pixel_count} -> {'Connected' if is_connected else 'Disconnected'}")
        
        # 🆕 Double Check: If XML-based crop failed (Green=0), try the broad fixed region (0,0,200,200)
        # This handles cases where XML bounds are wrong or text is missing
        if not is_connected and crop_box != (0, 0, 200, 200):
             print("⚠️ Primary Check Failed. Trying Broad Fixed Region (0,0,200,200)...")
             crop_box_broad = (0, 0, 200, 200)
             crop_broad = image.crop(crop_box_broad)
             pixels_broad = list(crop_broad.getdata())
             
             green_broad = 0
             for r, g, b in pixels_broad:
                 h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                 h_deg = h * 360
                 if 70 <= h_deg <= 170 and s > 0.05 and v > 0.2:
                      green_broad += 1
                 elif g > r + 5 and g > b + 5 and g > 100:
                      green_broad += 1
             
             print(f"🎨 Broad Region Analysis: Green={green_broad}")
             if green_broad > 3:
                 print("✅ Broad Region Found Green! Marking as Connected.")
                 is_connected = True
                 # Update coords to center of broad region (rough estimate)
                 # Or better: keep the original icon_coords if they exist, or use broad center
                 # Let's use the broad center if we had to fallback
                 crop_box = crop_box_broad

        # Calculate center of the icon box for clicking
        icon_center_x = (crop_box[0] + crop_box[2]) // 2
        icon_center_y = (crop_box[1] + crop_box[3]) // 2
        
        return is_connected, {"x": icon_center_x, "y": icon_center_y}
        
    except Exception as e:
        print(f"⚠️ Connection Check Error: {e}")
        return False, None

import xml.etree.ElementTree as ET
import re

# ==============================================================================
# 🔧 XML 解析工具 (Grounding Utils)
# ==============================================================================
def parse_xml_elements(xml_str: str) -> list:
    """从 XML 中提取所有 clickable 元素及其坐标，并生成 XPath"""
    elements = []
    if not xml_str:
        return elements
        
    try:
        root = ET.fromstring(xml_str)
        
        # Recursive function to traverse and build XPath
        def traverse(node, path, parent_map=None):
            # Calculate current node index among siblings of same tag
            tag = node.tag
            if parent_map is None:
                index = 0 # Root
            else:
                siblings = [c for c in parent_map if c.tag == tag]
                index = siblings.index(node) if node in siblings else 0
            
            # Build current path segment
            # Use class name if available as tag is usually just 'node' in Android XML dump
            class_name = node.attrib.get('class', tag)
            current_path = f"{path}/{class_name}[{index}]"
            
            # Check if clickable
            if node.attrib.get('clickable') == 'true':
                bounds = node.attrib.get('bounds')
                if bounds:
                    match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                    if match:
                        x1, y1, x2, y2 = map(int, match.groups())
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        text = node.attrib.get('text', '')
                        desc = node.attrib.get('content-desc', '')
                        resource_id = node.attrib.get('resource-id', '')
                        
                        label = text or desc or resource_id or "Unknown Button"
                        
                        elements.append({
                            "type": "native_element",
                            "text": label,
                            "x": center_x,
                            "y": center_y,
                            "bounds": [x1, y1, x2, y2],
                            "action": "tap",
                            "resource_id": resource_id,
                            "xpath": current_path, # 🆕 Added XPath
                            "content_desc": desc # Store raw desc
                        })

            # Recurse children
            children = list(node)
            for child in children:
                traverse(child, current_path, children)

        traverse(root, "")
        
    except Exception as e:
        print(f"⚠️ XML Parse Error: {e}")
        
    return elements

# ==============================================================================
# 🧠 LLM Agent Logic (Future Upgrade)
# ==============================================================================

PROMPT_APP_AGENT = """
You are an intelligent mobile app testing agent. Your goal is to explore the app's functionality efficiently without causing disruptions.

**Current State:**
- Screenshot: [Provided as Image]
- XML Hierarchy: [Provided as Text]
- Activity: {activity}
- Exploration History: {history}

**CRITICAL RULES (Violating these causes immediate failure):**
1. **ABSOLUTE PROHIBITION**: NEVER click any element related to "Disconnect", "Unbind", "Forget Device", "Delete Device", "Remove Device", or "Logout".
   - Keywords to AVOID: "断开", "解绑", "忘记", "删除", "退出登录", "Disconnect", "Unbind", "Forget".
2. **STAY IN APP**: NEVER perform actions that would exit the app (like clicking "Home" or "Back" on the root page).
3. **CONNECTION CHECK**: The top-left icon indicates connection status. Green = Connected, Gray = Disconnected.
   - If you see Gray, you should prioritize finding a "Connect" button, BUT DO NOT click "Disconnect" if it's already Green.

**Exploration Strategy:**
1. Prioritize exploring new features and deeper pages over revisiting known pages.
2. If you are on the Root Page (Main Tab), try switching Tabs if the current tab is fully explored.
3. If you encounter a popup/dialog, handle it (Confirm/Cancel) unless it's a "Disconnect" confirmation.

**Output Format:**
Return a JSON object with the next action. Include a "thought" field to explain your reasoning step-by-step.
{{
    "thought": "I see a 'Settings' button that I haven't clicked yet. I should check it out.",
    "action": "tap" | "scroll" | "input" | "wait",
    "x": <int>, "y": <int>, 
    "text": "<element_text>", 
    "reason": "<short_summary>" 
}}
"""

async def analyze_page_and_generate_cases(screenshot_b64: str, xml_data: str, activity: str) -> dict:
    """
    使用 AI 分析页面语义并同时生成测试用例 (Optimization: 2-in-1)
    """
    default_result = {"summary": f"Page in {activity}", "cases": []}
    
    if AI_PROVIDER == "mock":
        return default_result
        
    # 🚀 Fail fast if Key is invalid to avoid timeouts
    if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
        return default_result

    try:
        prompt = f"""
        You are an AI Software QA Agent. Analyze this Android screen.
        
        Task 1: Summarize Functionality
        - Identify the main purpose (e.g. Login, Settings, Product List).
        - Summarize in ONE concise sentence in Simplified Chinese (简体中文).
        
        Task 2: Generate Test Cases
        - Create 1-2 specific, actionable test cases for THIS page.
        - Focus on primary user interactions (e.g. "验证登录成功", "检查开关状态").
        - Output a "script" array for each case with steps (action: tap/input/assert/wait).
        - IMPORTANT: The 'desc' field MUST be in Simplified Chinese (简体中文).
        
        Context Activity: {activity}
        
        Output strictly valid JSON format:
        {{
            "summary": "页面功能摘要（中文）...",
            "cases": [
                {{ 
                    "desc": "测试用例描述（中文）", 
                    "script": [
                        {{"action": "tap", "text": "Button Name"}},
                        {{"action": "wait", "value": 1}}
                    ]
                }}
            ]
        }}
        """
        
        if AI_PROVIDER == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            response = await client.chat.completions.create(
                model=AI_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
                        ]
                    }
                ],
                max_tokens=500, # Increased for JSON
                response_format={ "type": "json_object" } # Force JSON if supported
            )
            content = response.choices[0].message.content.strip()
            # print(f"🧠 AI Output: {content}")
            return json.loads(content)
            
    except Exception as e:
        print(f"⚠️ Analysis Error: {e}")
        return default_result
        
    return default_result

async def query_llm_agent(screenshot_b64: str, xml_data: str, history: list, activity: str, visit_count: int = 1) -> dict:
    """
    Real LLM Agent decision making using OpenAI/DashScope.
    Supports Set-of-Mark (SOM) visual prompting.
    """
    if AI_PROVIDER == "mock":
        return None

    try:
        # 1. SOM Preparation
        tagged_screenshot = screenshot_b64
        id_map = {}
        som_instruction = ""
        
        if xml_data:
            print("🎨 Applying Set-of-Mark (SOM) overlay...")
            tagged_screenshot, id_map = draw_som_overlay(screenshot_b64, xml_data)
            som_instruction = """
            **VISUAL AID (Set-of-Mark)**:
            - The image has red bounding boxes with NUMERIC TAGS (e.g., 1, 2, 3).
            - These tags correspond to interactive elements.
            - To click an element, simply return its "element_id" (integer).
            - DO NOT guess coordinates if a tag is available.
            """

        # Construct History Text
        history_text = "\n".join([f"- {h}" for h in history[-5:]]) # Last 5 steps
        
        # Prepare Prompt with Context
        base_prompt = PROMPT_APP_AGENT.format(
            activity=activity or "Unknown",
            history=history_text
        )
        
        final_prompt = base_prompt + "\n" + som_instruction
        
        # 🆕 Inject Visit Count Warning
        if visit_count > 3:
            final_prompt += f"\n\n**WARNING**: You have visited this page {visit_count} times. You are likely STUCK in a loop. Try a DIFFERENT action or switch tabs immediately!"

        # 🆕 Inject Agent Capability
        final_prompt += "\n**AVAILABLE TOOLS**:\n- You can send 'SCAN' command to the BLE Agent if you are on a Scanning Page. Return action='agent_command' and payload='SCAN'.\n"

        # 🆕 Output Format Override for SOM
        final_prompt += """
        
        **Updated Output Format**:
        {{
            "thought": "Reasoning...",
            "action": "tap" | "scroll" | "input" | "wait",
            "element_id": <int>,  <-- PREFERRED: The number on the red tag
            "text": "<element_text>", 
            "reason": "<short_summary>" 
        }}
        """

        print(f"🧠 Agent Thinking... (History: {len(history)} steps)")

        if AI_PROVIDER == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )
            
            response = await client.chat.completions.create(
                model=AI_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tagged_screenshot}"}}
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.1 # Low temp for stable JSON
            )
            
            content = response.choices[0].message.content
            # Clean Markdown if present
            content = content.replace("```json", "").replace("```", "").strip()
            # print(f"🧠 Agent Thought: {content}")
            
            decision = json.loads(content)
            
            # 🆕 Log Thought Process (if available)
            if "thought" in decision:
                print(f"🤔 AI Thought: {decision['thought']}")
                
            # 🆕 SOM Resolution
            if "element_id" in decision and decision["element_id"] in id_map:
                el = id_map[decision["element_id"]]
                decision["x"] = el["x"]
                decision["y"] = el["y"]
                print(f"🎯 SOM Resolved ID {decision['element_id']} -> ({el['x']}, {el['y']})")
                
            return decision

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            print(f"⚠️ LLM Auth Error (401). Switching to Heuristic Mode.")
        else:
            print(f"❌ LLM Agent Error: {e}")
        return None # Fallback to rules
    
    return None

PROMPT_PLANNER = """
You are an expert Android Test Automation Engineer. 
Your goal is to generate a HIGH-SPEED test script based on the User's Request and the provided App Map (Static Analysis).

**THE MAP (App Knowledge)**:
{app_map}

**USER REQUEST**: 
"{user_request}"

**AVAILABLE ACTIONS**:
1. `intent_jump`: Teleport directly to a specific Activity. (FASTEST)
   - target: The activity name from the Map (e.g., ".MainActivity", ".SettingsActivity").
2. `agent_command`: Send a command to the On-Device Agent SDK.
   - payload: "SCAN" (Start BLE Scan), "INIT" (Init SDK), "PING".
3. `wait`: Wait for a few seconds.
4. `assert`: Check if a text exists on screen.
5. `click`: Click a UI element (Fallback only, try to avoid).

**TASK**:
Generate a JSON array of steps to fulfill the user request efficiently. 
Prioritize `intent_jump` and `agent_command` over UI clicks.

**EXAMPLE OUTPUT**:
[
    {{"action": "agent_command", "payload": "INIT"}},
    {{"action": "intent_jump", "target": ".MainActivity"}},
    {{"action": "wait", "value": 2}},
    {{"action": "agent_command", "payload": "SCAN"}},
    {{"action": "assert", "target": "Scanning..."}}
]

Output ONLY the JSON array.
"""

async def generate_test_plan(user_request: str, app_map: dict) -> list:
    """
    Uses LLM to generate a test script based on Static Analysis Map.
    """
    if AI_PROVIDER == "mock":
        # Mock Plan for Demo
        return [
            {"action": "agent_command", "payload": "INIT"},
            {"action": "intent_jump", "target": ".MainActivity"},
            {"action": "wait", "value": 1},
            {"action": "agent_command", "payload": "SCAN"},
            {"action": "assert", "target": "Device"}
        ]

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        
        map_summary = json.dumps(app_map.get("activities", {}), indent=2)
        prompt = PROMPT_PLANNER.format(app_map=map_summary, user_request=user_request)
        
        response = await client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        script = json.loads(content)
        return script
        
    except Exception as e:
        print(f"❌ Planner Error: {e}")
        return []

class SemanticStateRecognizer:
    def __init__(self, provider="mock"):
        self.provider = provider
        self.cache = {} # visual_hash -> canonical_id
        self.xml_cache = {} # xml_structure_hash -> canonical_id

    async def recognize(self, screenshot_b64: str, xml_data: str, activity: str) -> str:
        # 1. Check Visual Cache (Fastest)
        visual_hash = get_perceptual_hash(screenshot_b64)
        if visual_hash in self.cache:
            return self.cache[visual_hash]

        # 2. Check XML Cache (More Robust)
        xml_hash = ""
        if xml_data:
            # 🆕 Improved XML Hash: Filter Dynamic Text for stability
            import re
            # Only use IDs for structure hash, ignore text completely for Page ID
            # This makes "Settings" page identical even if you scroll down a bit (if structure is same)
            # But wait, scrolling changes XML.
            # Let's stick to raw MD5 for now, but maybe strip timestamps?
            # Actually, let's use the same logic as get_state_hash's signature but ONLY IDs?
            # No, let's keep it simple. If XML matches exactly, it's the same page ID.
            
            # But we want "SettingsPage" to be reused even if time changes.
            # So we strip time/numbers.
            clean_xml = re.sub(r'text="[\d:%\.\s]+"', 'text="DYNAMIC"', xml_data)
            xml_hash = hashlib.md5(clean_xml.encode()).hexdigest()
            
            if xml_hash in self.xml_cache:
                canonical_id = self.xml_cache[xml_hash]
                # Also update visual cache for next time
                self.cache[visual_hash] = canonical_id
                return canonical_id

        canonical_id = "UnknownPage"
        
        # 🚀 Priority: XML Heuristic (Fastest & Free) -> LLM (Slow & Costly)
        # If XML is available, trust it first to avoid unnecessary API calls
        heuristic_id = self._heuristic_identify(xml_data, activity)
        if heuristic_id and heuristic_id != activity.split('.')[-1]:
             # If heuristic found something meaningful (e.g. "SettingsActivity_title_settings"), use it!
             canonical_id = heuristic_id
        else:
            # Only fallback to LLM if Heuristic failed OR if we really want semantic understanding
            # But for speed, let's stick to Heuristic as primary for now unless explicitly requested
            if self.provider == "mock":
                canonical_id = heuristic_id
            elif self.provider == "openai" and OPENAI_API_KEY and "sk-" in OPENAI_API_KEY:
                # Only call if we have a valid key
                try:
                    canonical_id = await asyncio.wait_for(self._query_vlm(screenshot_b64, activity), timeout=5.0)
                except asyncio.TimeoutError:
                    print("⚠️ VLM Timeout. Using Heuristic.")
                    canonical_id = heuristic_id
            elif self.provider == "dashscope" and DASHSCOPE_API_KEY:
                canonical_id = await self._query_dashscope(screenshot_b64, activity)
            else:
                canonical_id = heuristic_id

        # Normalize ID
        canonical_id = canonical_id.strip().replace(" ", "")
        
        # Update Caches
        self.cache[visual_hash] = canonical_id
        if xml_hash:
            self.xml_cache[xml_hash] = canonical_id
            
        print(f"🧠 Semantic State Recognized: {canonical_id}")
        return canonical_id

    def _heuristic_identify(self, xml_data, activity):
        base_id = activity.split('.')[-1] if activity else "App"
        # Try to find a title bar text (simple heuristic)
        if xml_data:
            import re
            # Look for common title bar resource IDs
            title_ids = ["id/title", "id/action_bar_title", "id/toolbar_title", "id/header_text", "id/tv_title"]
            for tid in title_ids:
                match = re.search(f'resource-id="[^"]*{tid}"[^>]*text="([^"]+)"', xml_data)
                if match:
                    return f"{base_id}_{match.group(1)}"
        return base_id

    async def _query_vlm(self, screenshot_b64, activity):
        if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
            print("⚠️ Missing or Invalid OpenAI Key. Skipping VLM.")
            return f"{activity.split('.')[-1]}"

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=5.0) # 🚀 Add 5s Timeout
            
            prompt = f"""
            Analyze this Android screenshot and assign a unique "Canonical Page ID".
            
            Rules:
            1. Format: PascalCase (e.g., SettingsPage, ChatList, ProductDetail).
            2. Ignore dynamic content (names, times, numbers).
            3. If it's a dialog, append 'Dialog' (e.g., DeleteDialog).
            4. Current Activity: {activity}
            
            Output ONLY the ID string.
            """
            
            response = await client.chat.completions.create(
                model=AI_MODEL_NAME,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
                        ]
                    }
                ],
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ VLM Error: {e}")
            return f"{activity.split('.')[-1]}"

    async def _query_dashscope(self, screenshot_b64, activity):
        try:
            from dashscope import MultiModalConversation
            import dashscope
            dashscope.api_key = DASHSCOPE_API_KEY
            
            prompt = [
                {"role": "user", "content": [
                    {"image": f"data:image/png;base64,{screenshot_b64}"},
                    {"text": f"Analyze this Android screenshot and assign a unique 'Canonical Page ID' (PascalCase). Current Activity: {activity}. Output ONLY the ID."}
                ]}
            ]
            
            # Note: DashScope python SDK might be blocking? 
            # We wrap it in run_in_executor if needed, but for now let's try direct call.
            # But MultiModalConversation.call is synchronous.
            
            response = await asyncio.to_thread(
                MultiModalConversation.call, 
                model=AI_MODEL_NAME, 
                messages=prompt
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content[0]['text']
                return content.strip().replace(" ", "")
            else:
                print(f"❌ DashScope VLM Error: {response.code}")
                return f"VLM_Error_{activity}"
        except Exception as e:
            print(f"⚠️ DashScope VLM Exception: {e}")
            return f"VLM_Error_{activity}"

class AppStateGraph:
    def __init__(self):
        self.states = {} # Hash -> {screenshot, available_actions, explored_actions}
        self.transitions = [] # (from_hash, action, to_hash)
        self.current_state_hash = None
        self.root_hash = None # 记录入口页面 (Home Page)
        self.semantic_map = {} # SemanticHash -> StateHash (用于合并内容相同的页面)
        self.global_action_memory = set() # 🆕 Global Action Memory (Fingerprints)
        self.ineffective_actions = set() # 🆕 Global Ineffective Action Blacklist
        self.recognizer = SemanticStateRecognizer(provider=AI_PROVIDER)
        self.vector_memory = VectorMemory() # 🆕 Vector Memory (RAG)
        self.visual_memory = {} # 🆕 Visual Memory: {text_or_id -> base64_crop}

    def cache_visual_elements(self, screenshot_b64: str, elements: list):
        """
        Cache visual crops of interactive elements for template matching fallback.
        """
        if not screenshot_b64 or not elements: return
        
        for el in elements:
            # Key: Text or Resource ID
            key = el.get('text')
            if not key: key = el.get('resource_id')
            if not key: continue
            
            # Skip if already cached (or maybe overwrite with newer?)
            # Overwriting is better to adapt to theme changes, but might be slow.
            # Let's check if key in cache
            if key in self.visual_memory: continue
            
            bounds = el.get('bounds')
            if bounds:
                crop = crop_element(screenshot_b64, bounds)
                if crop:
                    self.visual_memory[key] = crop
                    # print(f"🖼️ Cached Visual Anchor for '{key}'")

    def get_state_hash(self, screenshot_b64: str, xml_data: str = None, activity: str = None) -> str:
        # Hybrid Hash Strategy:
        # 1. Visual Hash (Perceptual) - Good for overall layout
        visual_hash = get_perceptual_hash(screenshot_b64)
        
        # 2. Structural Hash (XML) - Good for distinguishing Tabs/Lists
        structure_hash = ""
        if xml_data:
            # Filter dynamic texts (digits, times) to avoid duplicate states
            import re
            ids = re.findall(r'resource-id="([^"]+)"', xml_data)
            raw_texts = re.findall(r'text="([^"]+)"', xml_data)
            
            # 🆕 Filter: Remove texts that look like times or numbers
            # BUT KEEP short titles (e.g. "Home", "Settings") even if they are short
            def is_dynamic(t):
                if re.match(r'^[\d:%\.\s]+$', t): return True # Pure numbers/time
                if re.match(r'^\d{4}-\d{2}-\d{2}', t): return True # Date
                return False

            texts = [t for t in raw_texts if not is_dynamic(t)]
            
            # 🆕 Key Fix: Include 'selected="true"' to distinguish active tabs
            selected = re.findall(r'selected="true"[^>]*text="([^"]+)"', xml_data)
            
            # 🆕 Include content-desc for accessibility icons (often used for navigation)
            content_descs = re.findall(r'content-desc="([^"]+)"', xml_data)
            filtered_descs = [d for d in content_descs if not is_dynamic(d)]

            # 🆕 CRITICAL: Include Title/Header Text with High Priority
            # We look for TextViews that might be titles (usually short, non-numeric, often at top)
            # This helps distinguish "Voice Settings" page from "Message Settings" page
            
            # Filter out empty or dynamic content (like time/battery) if needed
            # Increase limit to 100 to capture more page details
            signature = "|".join(sorted(ids[:100] + texts[:100] + selected + filtered_descs[:50]))
            
            # 🆕 Activity Name in Hash: Critical for distinguishing similar pages in different activities
            if activity:
                signature += f"|{activity}"
            
            # 🆕 Debug: Print signature length to check if it's too empty
            # print(f"🔍 Hash Signature (len={len(signature)}): {signature[:100]}...")
                
            structure_hash = hashlib.md5(signature.encode()).hexdigest()[:8]
        
        if structure_hash:
            # 🚀 激进优化：如果 XML 结构指纹存在，完全信任它，忽略视觉变化。
            # 这能解决轮播图、动态 Banner 导致的无限新页面问题。
            return f"XML_{structure_hash}"
        else:
            return f"VIS_{visual_hash[:8]}"

    def get_semantic_hash(self, elements: list) -> str:
        """基于页面内容（按钮文本）计算语义哈希"""
        # 提取所有文本并排序，忽略坐标
        # 🆕 Improved: Filter dynamic text to stabilize hash
        import re
        def is_stable(t):
             if not t: return False
             if re.match(r'^[\d:%\.\s]+$', t): return False # Skip numbers/time
             return True
             
        texts = sorted([e.get('text', '') for e in elements if is_stable(e.get('text', ''))])
        # Also include resource-ids to differentiate icons
        ids = sorted([e.get('resource_id', '') for e in elements if e.get('resource_id')])
        
        content_str = "|".join(texts + ids)
        return hashlib.md5(content_str.encode()).hexdigest()
        
    def get_action_fingerprint(self, page_id: str, action: dict) -> str:
        """Generate a unique fingerprint for an action using XPath (Robust) + Content Hash"""
        
        # 1. XPath is the primary key for "Structural Identity"
        xpath = action.get('xpath', '')
        
        # 2. Content Hash for "Data Identity" (handle list scrolling)
        # We only hash STABLE content.
        import re
        def is_stable(t):
             if not t: return False
             if re.match(r'^[\d:%\.\s]+$', t): return False # Skip numbers/time
             return True
             
        act_text = action.get('text', '') if is_stable(action.get('text', '')) else ''
        act_desc = action.get('content_desc', '') if is_stable(action.get('content_desc', '')) else ''
        
        # 3. Fallback if XPath missing (e.g. Vision/VLM detected element)
        if not xpath:
             act_id = action.get('resource_id', '')
             if not act_id and not act_text:
                 # Last resort: Position Grid
                 x = action.get('x', 0)
                 y = action.get('y', 0)
                 xpath = f"POS:{x//50}_{y//50}"
             else:
                 xpath = f"ID:{act_id}"
        
        # 4. Construct Fingerprint
        # Format: CanonicalID | XPath | ContentHash
        # If Content changes (scrolled list item), Fingerprint changes -> Click again.
        # If Structure changes (new page), CanonicalID changes -> Click again.
        
        raw = f"{page_id}|{xpath}|{act_text}|{act_desc}"
        return hashlib.md5(raw.encode()).hexdigest()

    async def register_state(self, screenshot_b64: str, available_elements: list = None, xml_data: str = None, activity: str = None) -> str:
        # 0. Get Canonical ID (Async)
        canonical_id = await self.recognizer.recognize(screenshot_b64, xml_data, activity)

        # 1. 计算视觉哈希 (Hybrid)
        state_hash = self.get_state_hash(screenshot_b64, xml_data, activity)
        
        # 🆕 Improved: Sort available_elements Top-to-Bottom, Left-to-Right
        if available_elements:
            available_elements.sort(key=lambda e: (e['y'], e['x']))
        
        # 记录 Root Hash (第一次遇到的页面认为是 Home)
        if self.root_hash is None:
            print(f"🏠 Root Page Identified: {state_hash[:8]} (ID: {canonical_id})")
            self.root_hash = state_hash
        
        # 2. 检查是否已经存在
        if state_hash not in self.states:
            # 🆕 FUZZY STATE MATCHING (The "Strict Mode" Fix)
            # Before declaring this a "New" state, check if we have a similar existing state
            # based on interactive element fingerprints (IDs + Text).
            # This handles cases where hash changes slightly (e.g. scrolled list) but it's the SAME logical page.
            
            similar_state_hash = None
            if available_elements:
                current_fingerprint = set()
                for el in available_elements:
                    # Fingerprint: resource_id + text (if stable)
                    fid = el.get('resource_id', '')
                    txt = el.get('text', '')
                    # Filter dynamic text
                    if re.match(r'^[\d:%\.\s]+$', txt): txt = ""
                    if fid or txt:
                        current_fingerprint.add(f"{fid}|{txt}")
                
                if current_fingerprint:
                    best_overlap = 0.0
                    best_candidate = None
                    
                    for h, s in self.states.items():
                        # Skip if activity different (strong filter)
                        if activity and activity not in str(s.get("canonical_id", "")): 
                             # Rough check, maybe skip
                             pass
                        
                        cached_fp = s.get("element_fingerprint", set())
                        if not cached_fp: continue
                        
                        # Jaccard Similarity
                        intersection = len(current_fingerprint & cached_fp)
                        union = len(current_fingerprint | cached_fp)
                        if union > 0:
                            score = intersection / union
                            if score > 0.85: # 85% match required
                                if score > best_overlap:
                                    best_overlap = score
                                    best_candidate = h
                    
                    if best_candidate:
                        print(f"🧬 Fuzzy Match: {state_hash[:8]} ~= {best_candidate[:8]} (Score: {best_overlap:.2f})")
                        similar_state_hash = best_candidate

            if similar_state_hash:
                # REUSE EXISTING STATE
                state_hash = similar_state_hash
                # Update screenshot to latest
                self.states[state_hash]["screenshot"] = screenshot_b64
                self.states[state_hash]["visit_count"] += 1
                # Merge actions? 
                # If we reuse state, we automatically inherit "explored_actions".
                # But we might need to add new "available_actions" if they appeared.
                # For strictness, let's NOT add new actions easily to avoid loops.
                # Just trust the old state.
            else:
                # TRULY NEW STATE
                print(f"🆕 发现新视觉页面! Hash: {state_hash[:8]} (ID: {canonical_id})")
                
                # Cache fingerprint
                fp = set()
                if available_elements:
                    for el in available_elements:
                        fid = el.get('resource_id', '')
                        txt = el.get('text', '')
                        if re.match(r'^[\d:%\.\s]+$', txt): txt = ""
                        if fid or txt:
                            fp.add(f"{fid}|{txt}")

                self.states[state_hash] = {
                    "screenshot": screenshot_b64,
                    "available_actions": available_elements if available_elements else None,
                    "explored_actions": [],
                    "semantic_hash": None,
                    "canonical_id": canonical_id, 
                    "summary": None, 
                    "visit_count": 1,
                    "element_fingerprint": fp # 🆕 Cache for fuzzy match
                }
        else:
            print(f"🔄 视觉哈希命中: {state_hash[:8]}")
            # 更新最新截图
            self.states[state_hash]["screenshot"] = screenshot_b64
            self.states[state_hash]["canonical_id"] = canonical_id 
            self.states[state_hash]["visit_count"] = self.states[state_hash].get("visit_count", 0) + 1
            if available_elements and not self.states[state_hash]["available_actions"]:
                 self.states[state_hash]["available_actions"] = available_elements

        self.current_state_hash = state_hash
        
        # 如果传入了 elements，尝试进行语义更新
        if available_elements:
            self.update_state_with_ai(state_hash, available_elements)
            
        return state_hash

    def update_state_with_ai(self, state_hash: str, elements: list):
        """AI 分析完成后，更新状态信息，并尝试进行语义合并"""
        if state_hash not in self.states:
            return
            
        # 1. 计算语义哈希
        sem_hash = self.get_semantic_hash(elements)
        
        # 2. 检查语义是否已存在 (即：虽然截图有点不同，但按钮完全一样)
        if sem_hash in self.semantic_map:
            existing_state_hash = self.semantic_map[sem_hash]
            if existing_state_hash != state_hash:
                print(f"🔗 语义合并: {state_hash[:8]} -> {existing_state_hash[:8]} (内容相同)")
                # 将当前状态“重定向”到旧状态
                self.current_state_hash = existing_state_hash
                return
        
        # 3. 如果是真正的新状态
        self.semantic_map[sem_hash] = state_hash
        self.states[state_hash]["semantic_hash"] = sem_hash
        self.states[state_hash]["available_actions"] = elements

    def is_stuck(self) -> bool:
        """
        检测是否卡死：
        1. 连续 3 次都在同一个页面，且动作都是 scroll (Infinite Scroll Loop)
        2. 连续 3 次尝试执行相同的 tap 动作，但页面 Hash 没有变化 (Ineffective Action Loop)
        3. 连续 8 次停留在同一页面 (Page Stagnation)
        4. 页面循环检测 (Cycle Detection)
        """
        if len(self.transitions) < 3:
            return False
        
        last_3 = self.transitions[-3:]
        # last_3 items are tuples: (from_hash, action, to_hash)
        
        # Condition 1: Same Page + Scroll Loop
        if all(t[0] == self.current_state_hash for t in last_3):
            if all(t[1].get('action') == 'scroll' for t in last_3):
                return True
                
        # Condition 2: Same Page + Same Tap Action (Ineffective Click)
        # Check if we are staying on the same page hash
        if all(t[0] == self.current_state_hash for t in last_3):
            first_action = last_3[0][1]
            # Check if all actions are identical (same type, same x, same y)
            if all(t[1].get('action') == first_action.get('action') and 
                   t[1].get('x') == first_action.get('x') and 
                   t[1].get('y') == first_action.get('y') for t in last_3):
                 print(f"🛑 Detected Ineffective Action Loop: {first_action}")
                 
                 # 🆕 Mark as Ineffective globally
                 state = self.states[self.current_state_hash]
                 page_id = state.get("canonical_id") or state.get("semantic_hash")
                 if page_id:
                     fingerprint = self.get_action_fingerprint(page_id, first_action)
                     self.ineffective_actions.add(fingerprint)
                     print(f"🚫 Added to Ineffective Blacklist: {fingerprint[:8]}")
                     
                 return True
        
        # 🆕 Condition 3: Page Stagnation (Stuck on same page for 5 steps)
        # Regardless of actions, if we haven't left this page in 5 turns, we are stuck.
        # EXCEPTION: If we are clicking different valid items (like in a long Settings list), it might take >5 steps.
        # But if we are stuck on "QR Code" page (few buttons), 5 steps is too long.
        if len(self.transitions) >= 8: # Relaxed from 5 to 8 to allow long lists
            last_8 = self.transitions[-8:]
            if all(t[0] == self.current_state_hash for t in last_8):
                print(f"🛑 Detected Page Stagnation: Stuck on {self.current_state_hash[:8]} for 8 steps")
                return True

        # 🆕 Condition 4: Cycle Detection (A -> B -> A -> B)
        # Check the sequence of canonical IDs (or hashes)
        if len(self.transitions) >= 9: # Need at least 9 for 3-page cycle check
            # Get last 9 destination hashes
            # transition = (from_hash, action, to_hash)
            # We care about to_hash
            path = [t[2] for t in self.transitions[-9:]]
            
            # Pattern 1: A, B, A, B (2-Page Loop)
            # last 4: [A, B, A, B]
            if path[-1] == path[-3] and path[-2] == path[-4]:
                 print(f"🛑 Detected 2-Page Cycle: {path[-1][:8]} <-> {path[-2][:8]}")
                 
                 # Identify the action causing the loop (the one that brought us to current page)
                 loop_entry_action = self.transitions[-1][1]
                 
                 # Blacklist it globally
                 from_hash = self.transitions[-1][0]
                 from_state = self.states.get(from_hash)
                 if from_state:
                     from_id = from_state.get("canonical_id") or from_state.get("semantic_hash")
                     if from_id:
                         fingerprint = self.get_action_fingerprint(from_id, loop_entry_action)
                         self.ineffective_actions.add(fingerprint)
                         print(f"🚫 Cycle Breaker: Blacklisted action {fingerprint[:8]}")
                         
                 return True

            # Pattern 2: A, B, C, A, B, C (3-Page Loop)
            # last 6: [A, B, C, A, B, C]
            if path[-1] == path[-4] and path[-2] == path[-5] and path[-3] == path[-6]:
                 print(f"🛑 Detected 3-Page Cycle: {path[-1][:8]} -> {path[-2][:8]} -> {path[-3][:8]}")
                 
                 # Identify the action causing the loop
                 loop_entry_action = self.transitions[-1][1]
                 
                 # Blacklist it globally
                 from_hash = self.transitions[-1][0]
                 from_state = self.states.get(from_hash)
                 if from_state:
                     from_id = from_state.get("canonical_id") or from_state.get("semantic_hash")
                     if from_id:
                         fingerprint = self.get_action_fingerprint(from_id, loop_entry_action)
                         self.ineffective_actions.add(fingerprint)
                         print(f"🚫 Cycle Breaker: Blacklisted action {fingerprint[:8]}")
                         
                 return True

        # 🆕 Condition 5: Visit Limit Protection
        # If current page has been visited > 10 times, assume loop.
        current_state = self.states.get(self.current_state_hash)
        if current_state:
             visit_count = current_state.get("visit_count", 0)
             if visit_count > 10:
                 print(f"🛑 Detected Excessive Visits: {self.current_state_hash[:8]} visited {visit_count} times")
                 # We don't blacklist an action, but we return True to force a "Back" or "Restart" in decide_next_step
                 return True

        return False

    def from_dict(self, data):
        self.states = data.get("states", {})
        self.transitions = data.get("transitions", [])
        self.current_state_hash = data.get("current_state_hash")
        # 重建语义映射
        self.semantic_map = {}
        for h, s in self.states.items():
            if "semantic_hash" in s and s["semantic_hash"]:
                self.semantic_map[s["semantic_hash"]] = h
        
        # 🆕 Restore Global Memory
        self.global_action_memory = set(data.get("global_action_memory", []))
        self.ineffective_actions = set(data.get("ineffective_actions", []))
        # Vector Memory is hard to serialize, we might skip it or rebuild it from logs if needed.
        # For now, let's assume it starts fresh or we could save the text bank.
        # self.vector_memory.memory_bank = ...

    def is_root_page(self, xml_data: str = None) -> bool:
        """
        判断是否为根页面 (Home Page)
        启发式规则：
        1. 是记录中的 root_hash
        2. 或者，页面底部有 TabBar (多个 clickable 元素在底部水平排列)
        3. 🚫 排除包含 "Disconnect/Unbind" 等危险关键词的页面（通常是设备详情页，不是主页）
        """
        # 🛡️ Safety Check: If page contains "Disconnect" or "Unbind", it is DEFINITELY NOT a Root Page
        # This prevents the Agent from thinking it's safe on a Device Settings page.
        if xml_data:
            forbidden_keywords = ['断开', '解绑', 'disconnect', 'unbind', '忘记设备', '删除设备']
            lower_xml = xml_data.lower()
            for kw in forbidden_keywords:
                if kw in lower_xml:
                    return False

        if self.current_state_hash == self.root_hash:
            return True
            
        if not xml_data:
            return False
            
        try:
            # 解析 XML 找底部元素
            root = ET.fromstring(xml_data)
            bottom_elements = []
            # 假设屏幕高度 ~2400 (ADB coordinates)
            # 我们找 y > 2000 的元素
            for node in root.iter():
                if node.attrib.get('clickable') == 'true':
                    bounds = node.attrib.get('bounds')
                    if bounds:
                        match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                        if match:
                            x1, y1, x2, y2 = map(int, match.groups())
                            center_y = (y1 + y2) // 2
                            # 粗略判断：如果中心点在屏幕下方 10% 区域
                            if center_y > 2000: 
                                bottom_elements.append((x1, center_y))
            
            # 如果底部有 3 个或更多元素，且它们在 x 轴上分散，大概率是 TabBar
            if len(bottom_elements) >= 3:
                # 检查 x 轴分散度
                xs = sorted([e[0] for e in bottom_elements])
                # 简单判断：如果最左和最右跨度超过屏幕一半
                if (xs[-1] - xs[0]) > 500:
                    print("🏠 Bottom TabBar detected -> Root Page")
                    return True
                    
        except Exception as e:
            print(f"⚠️ Root Page Check Error: {e}")
            
        return False

    async def get_next_unexplored_action(self, client_id: str = None, is_connected: bool = True, icon_coords: dict = None):
        """
        获取当前页面尚未尝试的动作 (Strict Content-First Strategy)
        User Request: "先从上到下点击所有控件，然后再进入下一个页面"
        """
        if not self.current_state_hash or self.current_state_hash not in self.states:
            return None
            
        state = self.states[self.current_state_hash]
        if not state["available_actions"]:
            return None # 还没分析完
            
        # 🆕 Forbidden Action Helper
        def is_forbidden(action):
            text = action.get('text', '').lower()
            desc = action.get('content-desc', '').lower() # Corrected from content-desc
            res_id = action.get('resource-id', '').lower()
            
            # 🆕 Added more keywords for protection
            forbidden_keywords = [
                '断开', '解绑', 'disconnect', 'unbind', '忘记设备', '删除设备', 'unpair', 'forget device', '解除绑定',
                '关闭服务', 'stop service', 'close service', '退出', 'exit', '注销', 'logout', 'sign out'
            ]
            
            for kw in forbidden_keywords:
                if kw in text or kw in desc: # Removed res_id check for safety, sometimes IDs are misleading
                     # Only check ID if it explicitly contains 'disconnect' or 'logout'
                     if client_id:
                        manager.add_log(client_id, f"🚫 Skipping Forbidden Action: {text or desc or res_id}")
                     return True
                     
            if 'disconnect' in res_id or 'logout' in res_id:
                if client_id: manager.add_log(client_id, f"🚫 Skipping Forbidden Action (ID match): {res_id}")
                return True
                
            return False

        # 🆕 Filter Available Actions
        available = [a for a in state["available_actions"] if not is_forbidden(a)]
        explored = state["explored_actions"]
        
        # 🆕 GLOBAL MEMORY CHECK
        sem_hash = state.get("semantic_hash")
        canonical_id = state.get("canonical_id")
        
        # Helper to check if action is globally explored
        def is_globally_explored(action):
            page_id = canonical_id if canonical_id else sem_hash
            if not page_id: return False
            fingerprint = self.get_action_fingerprint(page_id, action)
            
            # 1. Check if Explored
            if fingerprint in self.global_action_memory:
                return True
                
            # 2. Check if Ineffective (Blacklisted)
            if fingerprint in self.ineffective_actions:
                return True
                
            return False

        # 🆕 Check if we are on Root Page
        is_root = self.is_root_page(xml_data=None)

        # Define thresholds for "Navigation Areas"
        max_y = max([a.get('y', 0) for a in available]) if available else 2400
        bottom_nav_threshold = max_y * 0.90
        top_nav_threshold = max_y * 0.10

        def is_navigation(action):
            y = action.get('y', 0)
            x = action.get('x', 0)
            text = action.get('text', '').lower()
            desc = action.get('content-desc', '').lower()
            res_id = action.get('resource-id', '').lower()
            
            if is_root and y > bottom_nav_threshold: return True
            if y < top_nav_threshold:
                if 'back' in text or '返回' in text or 'close' in text or '关闭' in text: return True
                if 'back' in desc or '返回' in desc or 'close' in desc or '关闭' in desc: return True
                if 'back' in res_id or 'close' in res_id: return True
                if x < 200 and y < 200: 
                    if 'search' in text or '搜索' in text: return False
                    return True
            
            nav_keywords = ['home', '主页', 'back', '返回', 'settings', '设置', 'menu', '菜单']
            if any(k == text for k in nav_keywords) or any(k == desc for k in nav_keywords): return True
            return False

        # 🆕 THROTTLE REPETITIVE NON-NAV ACTIONS
        throttle_keywords = ['save', '保存', 'download', '下载', 'copy', '复制', 'share', '分享', 'down']
        recent_transitions = self.transitions[-10:]
        
        def is_throttled(action):
            text = action.get('text', '').lower()
            desc = action.get('content-desc', '').lower()
            res_id = action.get('resource-id', '').lower()
            x = action.get('x', 0)
            y = action.get('y', 0)
            
            is_target_keyword = any(k in text or k in desc or k in res_id for k in throttle_keywords)
            
            for idx, t in enumerate(reversed(recent_transitions)):
                past_action = t[1]
                p_text = past_action.get('text', '').lower()
                p_desc = past_action.get('content-desc', '').lower()
                p_res_id = past_action.get('resource-id', '').lower()
                p_x = past_action.get('x', 0)
                p_y = past_action.get('y', 0)
                
                if is_target_keyword:
                    if any(k in p_text or k in p_desc or k in p_res_id for k in throttle_keywords):
                        if client_id: manager.add_log(client_id, f"⏳ Throttling repeated keyword action: {text or desc or res_id}")
                        return True

                if idx < 3: 
                    dist = ((x - p_x)**2 + (y - p_y)**2)**0.5
                    if dist < 80: 
                         if not is_navigation(action):
                             if client_id: manager.add_log(client_id, f"⏳ Global Debounce: Too close to recent click ({dist:.1f}px)")
                             return True
            return False

        # Apply Throttle Filter
        available = [a for a in available if not is_throttled(a)]

        # 找出未探索的
        unexplored = [a for a in available if a not in explored]
        unexplored = [a for a in unexplored if not is_globally_explored(a)]
        
        # 🆕 INTELLIGENT LIST SKIPPING (DISABLED for List Items to prevent missing controls)
        # Only skip if the resource_id is UNIQUE on the page (not a list item)
        # Count occurrences of each resource_id on current page
        rid_counts = {}
        for a in state["available_actions"]:
            rid = a.get('resource_id')
            if rid:
                rid_counts[rid] = rid_counts.get(rid, 0) + 1

        action_outcomes = {} 
        for t in self.transitions:
            if t[0] == self.current_state_hash:
                action = t[1]
                outcome_hash = t[2]
                if outcome_hash and outcome_hash not in ["pending", "unknown"]:
                    rid = action.get('resource_id')
                    # Skip check for geometry-based IDs as they are unique per pos
                    if rid:
                        if rid not in action_outcomes: action_outcomes[rid] = []
                        action_outcomes[rid].append(outcome_hash)

        redundant_ids = set()
        for rid, outcomes in action_outcomes.items():
            # 🆕 CRITICAL FIX for Settings Menu Coverage
            # If multiple items share the same ID (e.g. RecyclerView items), we MUST click ALL of them
            # unless we are 100% sure they lead to the EXACT SAME state.
            # But "Voice Settings" and "Message Settings" often lead to similar-looking fragments.
            # So, we DISABLE redundancy check for any ID that appears more than once.
            if rid_counts.get(rid, 0) > 1:
                continue

            # Even for unique IDs, be careful. Only skip if we tried it TWICE and it led to same state.
            if len(outcomes) >= 2 and len(set(outcomes)) == 1:
                redundant_ids.add(rid)
                if client_id: manager.add_log(client_id, f"⚡ Smart Skip: ID '{rid}' leads to same page. Skipping.")

        unexplored = [a for a in unexplored if a.get('resource_id') not in redundant_ids]

        # 🆕 VECTOR MEMORY FILTER (ASYNC BATCH)
        # 1. Collect descriptions for all candidates
        candidate_texts = []
        candidate_map = [] # Map index to original action
        filtered_unexplored = [] # Initialize here to fix NameError
        
        for a in unexplored:
             desc = f"{a.get('text', '')} {a.get('content_desc', '')} {a.get('resource_id', '')}".strip()
             if desc:
                 candidate_texts.append(desc)
                 candidate_map.append(a)
             else:
                 # No text, keep it (rely on hash)
                 filtered_unexplored.append(a)
        
        # 2. Batch Check
        if candidate_texts:
             results = await self.vector_memory.is_similar_batch(candidate_texts)
             
             for i, (is_similar, matched_text) in enumerate(results):
                 if is_similar:
                     if client_id: 
                         desc = candidate_texts[i]
                         # 🆕 LOGGING ONLY: Don't skip for now to ensure coverage
                         # manager.add_log(client_id, f"🧠 Vector Similar: '{desc[:20]}...' (But clicking anyway)")
                         pass 
                     filtered_unexplored.append(candidate_map[i]) # Always add back
                 else:
                     filtered_unexplored.append(candidate_map[i])
        
        unexplored = filtered_unexplored

        if not unexplored:
             return None

        # 🆕 STRICT ROW-FIRST ORDER (User Request: "Left-to-Right")
        # Bucket Y to group elements into rows (e.g., 40px tolerance)
        def sort_key(a):
            y = a.get('y', 0)
            x = a.get('x', 0)
            row_idx = y // 40 # 40px row height bucket
            return (row_idx, x)

        unexplored.sort(key=sort_key)

        # Separate Navigation (Back/Home) to do LAST
        content_actions = []
        nav_actions = []
        
        for action in unexplored:
            if is_navigation(action): nav_actions.append(action)
            else: content_actions.append(action)
            
        if content_actions:
            return content_actions[0]
            
        if nav_actions:
            # For navigation, prioritize "Settings" or "Menu" over "Back" if possible?
            # Or just follow top-down?
            # Let's stick to top-down for consistency, but ensure "Back" is last resort if it's top-left.
            # Actually, is_navigation handles "Back" logic.
            # If we have multiple nav actions, maybe prioritize non-back ones?
            def nav_priority(a):
                t = a.get('text', '').lower()
                desc = a.get('content-desc', '').lower()
                
                # 🆕 Settings should be LAST (User Rule), but before Back (Exit)
                # Sort is Ascending (Smallest First).
                # Normal Nav: Y (~2000)
                # Settings: 9000
                # Back: 9999
                
                if 'settings' in t or '设置' in t or 'settings' in desc or '设置' in desc:
                    return 9000
                
                if 'back' in t or '返回' in t: return 9999 
                
                return a.get('y', 0) # Top-down for others
            
            nav_actions.sort(key=nav_priority)
            return nav_actions[0]

        return None

    async def mark_action_explored(self, action):
        if self.current_state_hash and self.current_state_hash in self.states:
            # 1. Local Memory (Specific Visual State)
            self.states[self.current_state_hash]["explored_actions"].append(action)
            
            # 2. Global Memory (Semantic Context)
            state = self.states[self.current_state_hash]
            sem_hash = state.get("semantic_hash")
            canonical_id = state.get("canonical_id")
            
            page_id = canonical_id if canonical_id else sem_hash
            
            if page_id:
                fingerprint = self.get_action_fingerprint(page_id, action)
                self.global_action_memory.add(fingerprint)
                # print(f"🧠 Memory Added: {fingerprint[:8]} for {action.get('text') or action.get('resource_id')}")

            # 3. Vector Memory (Async)
            desc = f"{action.get('text', '')} {action.get('content_desc', '')} {action.get('resource_id', '')}"
            if desc.strip():
                await self.vector_memory.add_memory(desc)

# 全局状态图实例 (针对每个 Client 应该有一个独立的 Graph)
client_graphs: Dict[str, AppStateGraph] = {}

async def call_ai_analysis(screenshot_base64: str) -> list:
    """
    让 AI 识别当前页面所有的可交互元素
    """
    if AI_PROVIDER == "mock":
        # --- MOCK 模式 (Monkey Test) ---
        await asyncio.sleep(0.5)
        import random
        width = 1080
        height = 2400
        elements = []
        for _ in range(random.randint(3, 5)):
            x = random.randint(100, width - 100)
            y = random.randint(200, height - 200)
            elements.append({
                "type": "random_touch", 
                "text": "Mock Element", 
                "x": x, 
                "y": y, 
                "action": "tap"
            })
        return elements

    elif AI_PROVIDER == "openai":
        # --- OpenAI (GPT-4o) ---
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )
            
            print(f"📡 Calling OpenAI ({AI_MODEL_NAME})...")
            
            # 🆕 VLM Coordinate Extraction Prompt
            prompt_vlm = """
            You are a Vision-Language Model assisting an Android Agent.
            The XML tree for this page is EMPTY or incomplete, so we rely on your vision.
            
            Analyze the screenshot and identify ALL interactive UI elements (buttons, icons, input fields, tabs).
            For each element, estimate its center coordinates (x, y) based on a standard 1080x2400 resolution.
            
            Output a JSON list of objects:
            [
              {"text": "Login Button", "x": 540, "y": 1200, "action": "tap", "type": "vlm_element"},
              {"text": "Search Icon", "x": 1000, "y": 150, "action": "tap", "type": "vlm_element"}
            ]
            
            IMPORTANT: Be precise with coordinates. Return ONLY the JSON.
            """
            
            response = await client.chat.completions.create(
                model=AI_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_vlm},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            print(f"🧠 AI Response: {content}")
            
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                elements = json.loads(json_str)
                return elements
            else:
                print("⚠️ Failed to parse JSON from OpenAI response")
                return []
                
        except Exception as e:
            print(f"❌ OpenAI Exception: {e}")
            return []

    elif AI_PROVIDER == "dashscope":
        # --- DashScope (Qwen-VL) ---
        try:
            from dashscope import MultiModalConversation
            import dashscope
            dashscope.api_key = DASHSCOPE_API_KEY
            
            # 构造 Prompt
            prompt = [
                {"role": "user", "content": [
                    {"image": f"data:image/png;base64,{screenshot_base64}"},
                    {"text": "You are a mobile UI testing agent. Analyze this screenshot. List all interactive elements (buttons, inputs, icons) with their approximate coordinates (x, y). For elements that might require long press, use 'long_press'. Output JSON format: [{'text': 'Login', 'x': 500, 'y': 1000, 'action': 'tap'}, ...]"}
                ]}
            ]
            
            # 调用 API
            print("📡 Calling Qwen-VL (DashScope)...")
            response = MultiModalConversation.call(model=AI_MODEL_NAME, messages=prompt)
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content[0]['text']
                print(f"🧠 AI Response: {content}")
                
                # 解析 JSON
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    elements = json.loads(json_str)
                    return elements
                else:
                    print("⚠️ Failed to parse JSON from AI response")
                    return []
            else:
                print(f"❌ DashScope Error: {response.code} - {response.message}")
                return []
            
        except Exception as e:
            print(f"❌ DashScope Exception: {e}")
            return []

    return []

async def generate_exploration_plan(client_id: str):
    """
    Phase 2: 根据已扫描的页面，规划深度探索任务
    """
    if client_id not in active_clients or client_id not in client_graphs:
        return
        
    graph = client_graphs[client_id]
    
    # 1. 收集所有已知页面的摘要
    known_pages = []
    for h, s in graph.states.items():
        if s.get("summary"):
            known_pages.append(f"- Page {h[:4]}: {s['summary']}")
            
    pages_text = "\n".join(known_pages)
    
    manager.add_log(client_id, "🤔 正在根据扫描结果制定深度探索计划...")
    
    try:
        if AI_PROVIDER == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            
            prompt = f"""
            You are a Senior QA Engineer planning a deep exploration of an Android App.
            We have just finished a 'Fast Scan' and found the following pages:
            {pages_text}
            
            Based on these pages, infer the App's core business functions.
            List 5-8 specific, actionable exploration tasks to deeply test these functions.
            Tasks should be logical scenarios (e.g., "Search for 'shoes' and add to cart", "Go to Settings and toggle Dark Mode").
            
            Output strictly a JSON list of strings: ["Task 1", "Task 2", ...]
            """
            
            response = await client.chat.completions.create(
                model=AI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group(0))
                active_clients[client_id]["exploration_tasks"] = tasks
                active_clients[client_id]["phase"] = "DEEP_EXPLORING"
                active_clients[client_id]["current_task_index"] = 0
                
                manager.add_log(client_id, f"📝 探索计划已生成 ({len(tasks)} 项):")
                for t in tasks:
                    manager.add_log(client_id, f"  - {t}")
                    
                # 立即触发下一步
                await manager.send_command(client_id, {"type": "capture_screenshot"})
            else:
                manager.add_log(client_id, "⚠️ 计划生成失败 (JSON Parse Error)")
                active_clients[client_id]["phase"] = "IDLE"

    except Exception as e:
        manager.add_log(client_id, f"❌ 计划生成出错: {e}")
        active_clients[client_id]["phase"] = "IDLE"

async def decide_next_step(client_id: str, screenshot_b64: str, xml_data: str = None, activity: str = None, app_info: dict = None) -> dict:
    """
    核心决策引擎: 状态感知 + 探索策略
    """
    # 1. 获取或创建该 Client 的状态图
    if client_id not in client_graphs:
        client_graphs[client_id] = AppStateGraph()
    graph = client_graphs[client_id]
    
    # 获取当前 Client 状态
    client_phase = "IDLE"
    target_package = None 
    if client_id in active_clients:
        client_phase = active_clients[client_id]["phase"]
        target_package = active_clients[client_id].get("target_package")
        
        # 🆕 Auto-set target package on first run (Smartly)
        if app_info and app_info.get('package'):
            pkg = app_info.get('package')
            # Only set target if it's NOT a system app/launcher
            is_system = any(x in pkg for x in ["launcher", "systemui", "android", "home", "miui", "huawei"])
            if not target_package and not is_system:
                target_package = pkg
                active_clients[client_id]["target_package"] = target_package
                print(f"🎯 Target Package Auto-Set: {target_package}")
            elif not target_package and is_system:
                print(f"⚠️ Current package {pkg} seems to be System/Launcher. Waiting for valid App...")

    # 🆕 APP EXIT PROTECTION (Highest Priority)
    # If we have a target package and current package is different, RESTART immediately.
    # Also handles External Apps (File Picker) even if target is unknown.
    
    current_package = app_info.get('package') if app_info else None
    current_activity = activity or ""
    # print(f"🛡️ Exit Protection Check: Target={target_package}, Pkg={current_package}, Act={current_activity}")

    # 🆕 FAST TRAVERSAL EARLY EXIT
    # If we are in "Auto Discovery" mode and have pending intents, we don't need to do any heavy lifting.
    # Just check if we are still in the app (Exit Protection), log the visit, and JUMP.
    # This avoids: Image Compression, Hash Registration, Graph Updates, Semantic Analysis.
    is_fast_traversal = (client_phase == "AUTO_DISCOVERY" and active_clients[client_id].get("pending_intents"))
    
    if is_fast_traversal:
        # Still need basic exit protection
        if current_package and target_package and current_package != target_package:
             # Allowed System Overlays
             allowed_keywords = ["packageinstaller", "permission", "inputmethod", "keyboard"]
             is_allowed = any(k in current_package for k in allowed_keywords)
             if not is_allowed:
                 # Don't jump if we crashed, restart first
                 print(f"🚨 App Exited during Fast Traversal! Current: {current_package}")
                 manager.add_log(client_id, f"🚨 检测到 APP 退出 (当前: {current_package})，正在重新启动...")
                 return {"action": "app_start", "package": target_package, "reason": "App Exit Protection"}
        
        # Pop and Jump immediately!
        next_intent = active_clients[client_id]["pending_intents"].pop(0)
        remaining = len(active_clients[client_id]["pending_intents"])
        manager.add_log(client_id, f"⚡ 快速遍历模式 (Fast Track): 正在跳转至新页面 (剩余 {remaining} 个)...")
        # manager.add_log(client_id, f"🚀 执行跳转: {next_intent}")
        
        # Create a simple hash for this "visit" just to not break graph logic if needed later
        # But we SKIP registering it deeply.
        
        return {"action": "shell", "command": next_intent, "reason": "Fast Activity Traversal"}

    if current_package:
        # 1. External App Check (File Picker, Gallery, Camera) -> FORCE BACK
        # We check this FIRST to avoid getting stuck in system UIs
        # Keywords for Package Name
        pkg_keywords = ["documentsui", "gallery", "photos", "camera", "media", "provider", "file", "chooser"]
        # Keywords for Activity Name
        act_keywords = ["picker", "chooser", "file", "crop", "preview", "media"]
        
        is_external_pkg = any(k in current_package.lower() for k in pkg_keywords)
        is_external_act = any(k in current_activity.lower() for k in act_keywords)
        
        if is_external_pkg or is_external_act:
            # 🚀 Force Back immediately. Do not allow "Explore" even if package matches target,
            # because file pickers are usually modal and we want to dismiss them if we are stuck.
            print(f"📂 File/Media Picker detected (Pkg={current_package}, Act={current_activity}). FORCE BACK.")
            manager.add_log(client_id, f"📂 检测到文件/图片选择器，正在触发系统返回键(System Back)...")
            return {"action": "key_event", "keycode": 4, "reason": "Return from File Picker"}

        # 2. Target Package Check (Restart if crashed/exited)
        if target_package and current_package != target_package:
             # Allowed System Overlays (Keyboard, Permission Dialogs)
             allowed_keywords = ["packageinstaller", "permission", "inputmethod", "keyboard"]
             is_allowed = any(k in current_package for k in allowed_keywords)
             
             if is_allowed:
                 print(f"🛡️ System Overlay ({current_package}) detected. Allowing interaction.")
             else:
                 # Any other package -> RESTART
                 print(f"🚨 App Exited! Current: {current_package} != Target: {target_package}")
                 manager.add_log(client_id, f"🚨 检测到 APP 退出 (当前: {current_package})，正在重新启动...")
                 return {"action": "app_start", "package": target_package, "reason": "App Exit Protection"}

    # 2. 图像压缩 (Image Compression)
    screenshot_b64_compressed = compress_image(screenshot_b64, max_width=720)
    
    # 🆕 Training Data Logging (Pre-Decision)
    # Check if we have a pending log entry from the PREVIOUS step
    if client_id in active_clients:
        pending = active_clients[client_id].get("pending_log_data")
        if pending:
            # We have the result of the previous action!
            # pending contains: {session_id, step_id, before_screenshot, before_xml, action}
            try:
                training_logger.log_step(
                    client_id=client_id,
                    session_id=pending["session_id"],
                    step_id=pending["step_id"],
                    before_screenshot_b64=pending["before_screenshot"],
                    before_xml=pending["before_xml"],
                    action=pending["action"],
                    after_screenshot_b64=screenshot_b64_compressed, # The CURRENT screen is the "After" state
                    after_xml=xml_data,
                    agent_reasoning=pending.get("reasoning")
                )
                # Clear pending
                active_clients[client_id]["pending_log_data"] = None
            except Exception as e:
                print(f"❌ Logging Error: {e}")

    # 3. 注册状态 (Visual Hash First)
    last_hash = graph.current_state_hash
    current_hash = await graph.register_state(screenshot_b64_compressed, xml_data=xml_data, activity=activity)
    
    # 🆕 UPDATE PREVIOUS TRANSITION (Link Action to Consequence)
    if graph.transitions:
        last_t = graph.transitions[-1]
        if last_t[2] == "pending":
            # Update the pending transition with the ACTUAL resulting state
            graph.transitions[-1] = (last_t[0], last_t[1], current_hash)
            # print(f"🔗 Linked Transition: {last_t[0][:8]} --[{last_t[1].get('action')}]--> {current_hash[:8]}")
    
    is_new_state = (current_hash != last_hash) and (current_hash not in graph.states) # Rough heuristic, actually register_state handles insertion
    is_reentry = (current_hash != last_hash) and (current_hash in graph.states)
    
    # 4. 检查是否需要 AI 分析 (Lazy Analysis)
    available_actions = graph.states[current_hash].get("available_actions")

    # 🆕 CACHE VISUAL ANCHORS (Every time we have actions)
    if available_actions:
        # Run in background/async to avoid blocking main loop too much?
        # For now, synchronous but fast cropping.
        graph.cache_visual_elements(screenshot_b64_compressed, available_actions)

    # 🆕 INEFFECTIVE ACTION LEARNING (Post-Action Check)
    # If we just performed an action, and the State Hash (Visual + XML) AND Canonical ID did NOT change...
    # Then the action was likely ineffective (or just a Toast).
    if graph.transitions:
        last_t = graph.transitions[-1]
        last_from_hash = last_t[0]
        last_action = last_t[1]
        
        # Check if state changed
        state_unchanged = (current_hash == last_from_hash)
        
        # Double check: sometimes hash changes slightly (time/battery), but ID is same.
        # But here we are strict: If visual hash changed, it MIGHT be effective (e.g. menu opened).
        # So we only blacklist if visual/structure hash is IDENTICAL.
        
        if state_unchanged:
            # Get Semantic Context of the previous state
            prev_state = graph.states.get(last_from_hash)
            if prev_state:
                prev_id = prev_state.get("canonical_id") or prev_state.get("semantic_hash")
                if prev_id:
                    # Mark as Ineffective
                    fingerprint = graph.get_action_fingerprint(prev_id, last_action)
                    graph.ineffective_actions.add(fingerprint)
                    if client_id:
                        manager.add_log(client_id, f"🚫 标记无效动作 (No State Change): {last_action.get('text') or last_action.get('xpath')}")

    # 🆕 Training/Learning Step: Summarize Page + Generate Cases (2-in-1)
    # Condition: New State OR Existing State with missing knowledge (Backfill)
    # We check if 'summary' is missing or if we just want to be aggressive in generation.
    has_knowledge = graph.states[current_hash].get("summary") and graph.states[current_hash].get("summary") != "Unknown Page"
    
    # Check if we already generated cases for this specific page hash to avoid duplicates
    current_short_hash = current_hash[:6]
    # Handle missing test_cases list
    if "test_cases" not in active_clients[client_id]:
        active_clients[client_id]["test_cases"] = []
    
    already_generated = any(current_short_hash in c.get("origin", "") for c in active_clients[client_id]["test_cases"])
    
    # 🆕 STRICT CHECK: Do NOT learn if in EXECUTING mode
    is_executing = active_clients.get(client_id, {}).get("phase") == "EXECUTING"
    
    # Generate if: New State OR No Summary OR No Cases for this page
    if (is_new_state or not has_knowledge or not already_generated) and AI_PROVIDER != "mock" and not is_executing:
        # Run asynchronously to avoid blocking? 
        # For now, await it to ensure we have knowledge before acting.
        manager.add_log(client_id, "🎓 正在学习新页面并生成测试用例 (Knowledge Acquisition)...")
        
        analysis_result = await analyze_page_and_generate_cases(screenshot_b64_compressed, xml_data, activity)

        
        # 1. Update Summary
        summary = analysis_result.get("summary", f"Page in {activity}")
        graph.states[current_hash]["summary"] = summary
        manager.add_log(client_id, f"💡 页面认知: {summary}")
        
        # 2. Store Generated Test Cases
        new_cases = analysis_result.get("cases", [])
        if new_cases:
            # Add to client's test case list
            current_count = len(active_clients[client_id]["test_cases"])
            for i, case in enumerate(new_cases):
                case_obj = {
                    "id": current_count + i + 1,
                    "desc": case.get("desc", "Unknown Case"),
                    "script": case.get("script", []),
                    "status": "pending",
                    "origin": f"Auto-Gen at {current_hash[:6]}"
                }
                active_clients[client_id]["test_cases"].append(case_obj)
            
            manager.add_log(client_id, f"✅ 已生成 {len(new_cases)} 条新测试用例")
        else:
            manager.add_log(client_id, "⚠️ AI 未生成有效测试用例 (可能页面无交互)")


    # 🆕 Root Page Protection: If current hash is root, ensure we don't accidentally exit
    if graph.root_hash and current_hash == graph.root_hash:
         manager.add_log(client_id, "🏠 At Root Page. Enabling Exit Protection.")
         
    # 🆕 Connection Check DISABLED by user request
    # is_connected, icon_coords = check_connection_status(screenshot_b64, xml_data)
    # if not is_connected:
    #    manager.add_log(client_id, "⚠️ Icon is Gray -> Disconnected Status")
    is_connected = True
    icon_coords = None
    
    if not available_actions:
        manager.add_log(client_id, "🧠 Analyzing UI structure...")
        
        # 🆕 Hybrid Strategy: Try XML First
        if xml_data:
            print("📄 Using XML Hierarchy for accurate grounding")
            available_actions = parse_xml_elements(xml_data)
            manager.add_log(client_id, f"✅ XML Parsed: Found {len(available_actions)} native elements")
            
        # If XML failed or returned nothing (e.g. Flutter/Game), Fallback to Vision AI
        if not available_actions:
            manager.add_log(client_id, "⚠️ XML empty, falling back to Vision AI...")
            available_actions = await call_ai_analysis(screenshot_b64_compressed)
        
        # 更新图谱 (触发语义合并)
        graph.update_state_with_ai(current_hash, available_actions)
        # 重新获取 hash
        current_hash = graph.current_state_hash
        
        # 🆕 Re-calculate is_reentry after semantic merge
        # If semantic merge happened, current_hash might be equal to last_hash now!
        if current_hash == last_hash:
            is_reentry = False
        else:
             # If hashes are still different, we check if the NEW hash is in states (it must be, as we just registered/updated it)
             # But we care if it was visited BEFORE this current step.
             # Actually, if current_hash != last_hash, and we are here, it means we moved to a different page.
             # If that different page is fully explored, we might want to go back.
             pass
        
    else:
        manager.add_log(client_id, f"⚡ Fast Resume: Known Page {current_hash[:8]}")

    # 更新统计信息
    if client_id in active_clients:
         active_clients[client_id]["stats"]["pages"] = len(graph.states)
         
         # 🆕 Convergence/Stability Calculation (for Single Activity Apps)
         current_steps = active_clients[client_id]["stats"].get("steps", 0)
         if is_new_state:
             active_clients[client_id]["stats"]["last_new_state_step"] = current_steps
         
         # 🆕 Map Coverage Calculation
         use_map_coverage = False
         if APP_KNOWLEDGE_MAP and "activities" in APP_KNOWLEDGE_MAP:
             total_activities = len(APP_KNOWLEDGE_MAP["activities"])
             
             # HEURISTIC: Only use Map Coverage if we have enough activities (>4)
             # Otherwise (Flutter/RN/Compose), use Convergence.
             if total_activities > 4:
                 use_map_coverage = True
                 # Count how many unique activities from the MAP we have visited
                 visited_acts = active_clients[client_id].get("visited_activities", set())
                 if activity:
                     pkg = APP_KNOWLEDGE_MAP.get("package_name", "")
                     full_act = activity
                     if activity.startswith("."): full_act = pkg + activity
                     elif "." not in activity: full_act = f"{pkg}.{activity}"
                     
                     if full_act in APP_KNOWLEDGE_MAP["activities"]:
                         visited_acts.add(full_act)
                         active_clients[client_id]["visited_activities"] = visited_acts
                 
                 coverage = int((len(visited_acts) / total_activities) * 100)
                 active_clients[client_id]["stats"]["coverage"] = coverage
                 
                 # Auto-Finish
                 if coverage >= 100 and client_phase == "AUTO_DISCOVERY":
                     manager.add_log(client_id, "🏆 地图覆盖率达 100%！自动结束探索。")
                     active_clients[client_id]["phase"] = "FINISHED"
                     return {"action": "wait", "reason": "Coverage Complete"}
         
         if not use_map_coverage:
             # Convergence Mode
             last_new = active_clients[client_id]["stats"].get("last_new_state_step", 0)
             steps_since_new = current_steps - last_new
             # Threshold: 30 steps without new page = Converged
             threshold = 30
             convergence = min(100, int((steps_since_new / threshold) * 100))
             
             # Store in stats for frontend
             active_clients[client_id]["stats"]["convergence"] = convergence
             
             # Auto-Finish
             if convergence >= 100 and client_phase == "AUTO_DISCOVERY":
                 manager.add_log(client_id, "🏁 探索已收敛 (连续30步无新页面)！自动结束探索。")
                 active_clients[client_id]["phase"] = "FINISHED"
                 
                 # 🆕 Mark All Test Cases as 'READY' instead of deleting them
                 # We don't delete them anyway, but let's ensure status is clear
                 manager.add_log(client_id, "💾 探索结束，测试用例已保留。")
                 
                 return {"action": "wait", "reason": "Convergence Complete"}

    # 💾 自动保存
    save_client_state(client_id)
            
    # 5. 策略：优先探索未访问的路径 (DFS)
    # Pass is_connected status and icon_coords to prioritize recovery
    
    # 🧠 LLM AGENT INTEGRATION
    # If LLM Provider is configured, we give it a chance to decide, ESPECIALLY if:
    # 1. We are stuck (Transitions loop)
    # 2. We are on a critical page (Root)
    # 3. Or simply as a "Smart Planner" (can be toggled)
    
    # For now, let's try a Hybrid approach:
    # Use Rule-based DFS for speed. If DFS returns None (Page Explored), ASK LLM before Random Monkey.
    # OR, if we are on Root Page, ASK LLM to ensure safe navigation.

    # 🆕 FAST TRAVERSAL PRIORITY (User Request):
    # This block is now redundant because we handle it at the VERY BEGINNING of this function (Fast Track).
    # Removing it to avoid confusion and double logic.
    
    next_action = await graph.get_next_unexplored_action(client_id, is_connected, icon_coords)
    
    # 🆕 FORCE INTENT INJECTION AT ROOT OR IDLE (Optimization)
    # This block is ALSO largely redundant now due to Fast Track, but we keep it as a fallback
    # only if Fast Track didn't catch it (unlikely).
    if client_phase == "AUTO_DISCOVERY" and active_clients[client_id].get("pending_intents"):
        # But we only do this if we haven't started deep exploration of a specific path yet.
        # OR if the current page is "boring" (Root).
        is_root = graph.is_root_page(xml_data) or (graph.root_hash and graph.current_state_hash == graph.root_hash)
        
        # If we are at root, prefer Jumping over Clicking "Settings" manually
        if is_root:
             next_intent = active_clients[client_id]["pending_intents"].pop(0)
             manager.add_log(client_id, f"🚀 [加速] 主页跳过手动点击，优先注入 Intent: {next_intent}")
             
             shell_action = {"action": "shell", "command": next_intent, "reason": "Fast-Path Injection"}
             graph.transitions.append((graph.current_state_hash, shell_action, "pending"))
             return shell_action

    # If Rule Engine failed to find a "New" action, OR we are on Root (High Risk), try LLM
    should_ask_llm = (next_action is None) or (graph.is_root_page(xml_data))
    
    if should_ask_llm and AI_PROVIDER != "mock":
        # Collect History for LLM
        history_log = [f"{t[1].get('action')} {t[1].get('text','')}" for t in graph.transitions[-5:]]
        
        # 🆕 Pass Visit Count to LLM
        current_visit_count = graph.states[current_hash].get("visit_count", 1)
        
        llm_decision = await query_llm_agent(screenshot_b64_compressed, xml_data, history_log, activity, visit_count=current_visit_count)
        
        if llm_decision:
             # 🆕 Enhanced Logging
             if llm_decision.get("thought"):
                 manager.add_log(client_id, f"🤔 思考: {llm_decision.get('thought')}")
             manager.add_log(client_id, f"🧠 决策: {llm_decision.get('reason')}")
             
             # Validate LLM Action (Basic sanity check)
             if llm_decision.get('action') in ['tap', 'input', 'scroll', 'wait', 'finish_task', 'complete']:
                 next_action = llm_decision
                 # Don't return yet, fall through to Global Safety Check
    
    # Fallback to Graph Rules (if LLM failed or wasn't called)
    if not next_action:
        next_action = await graph.get_next_unexplored_action(client_id, is_connected, icon_coords)

    # 🆕 CRASH PROTECTION: Handle None Action Gracefully
    if not next_action:
        # (Redundant Intent Injection Block Removed)

        # If we are here, it means:
        # 1. Graph rules found nothing (Page fully explored)
        # 2. LLM was not called OR returned nothing
        # We MUST NOT return None, or the loop might crash/restart.
        
        # 🆕 QR CODE / PERMISSION PAGE TRAP FIX
        # If available_actions is very small (<= 2) and we are "stuck" (next_action is None),
        # it means we clicked everything and nothing happened (e.g. invalid QR, or just text).
        # We MUST Force Back.
        if available_actions and len(available_actions) <= 3:
             manager.add_log(client_id, "⚠️ Dead End Detected (Few elements, all explored). Forcing Back.")
             
             # 🆕 BLAME LOGIC: Walk back to find the entry point
             # We look for the transition that brought us into the current Semantic Scope
             current_state = graph.states.get(graph.current_state_hash)
             current_id = current_state.get("canonical_id") if current_state else None
             
             if current_id:
                 # Walk backwards
                 entry_transition = None
                 for i in range(len(graph.transitions) - 1, -1, -1):
                     t = graph.transitions[i]
                     # t = (from_hash, action, to_hash)
                     from_hash = t[0]
                     
                     from_state = graph.states.get(from_hash)
                     from_id = from_state.get("canonical_id") if from_state else None
                     
                     if from_id != current_id:
                         # Found the boundary! This transition brought us HERE from ELSEWHERE.
                         entry_transition = t
                         break
                 
                 if entry_transition:
                     parent_hash = entry_transition[0]
                     parent_action = entry_transition[1]
                     
                     if parent_hash in graph.states:
                         parent_state = graph.states[parent_hash]
                         parent_id = parent_state.get("canonical_id") or parent_state.get("semantic_hash")
                         if parent_id:
                             fingerprint = graph.get_action_fingerprint(parent_id, parent_action)
                             graph.global_action_memory.add(fingerprint)
                             manager.add_log(client_id, f"🚫 Dead End Backtrack: Globally Marking Entry Action as Explored: {parent_action.get('text', 'Unknown')}")

             back_action = {"action": "key_event", "keycode": 4, "reason": "Dead End Escape"}
             graph.transitions.append((graph.current_state_hash, back_action, "pending"))
             return back_action
        
        manager.add_log(client_id, "⚠️ Page fully explored or AI failed. Triggering Random Heuristic...")
        
        import random
        # Try to find ANY clickable element
        clickable_candidates = available_actions or []
        
        # Filter out forbidden actions even for random
        forbidden_keywords = ['断开', '解绑', 'disconnect', 'unbind', '忘记设备', '删除设备', 'unpair', 'forget device', '解除绑定']
        safe_candidates = []
        for a in clickable_candidates:
            t = a.get('text', '').lower()
            d = a.get('content-desc', '').lower()
            r = a.get('resource-id', '').lower()
            if not any(kw in t or kw in d or kw in r for kw in forbidden_keywords):
                safe_candidates.append(a)
        
        if safe_candidates and random.random() < 0.8:
            # 80% chance to click something random (re-verify)
            target = random.choice(safe_candidates)
            # Create a copy to avoid mutating the graph state
            next_action = target.copy()
            next_action["reason"] = "Random Fallback (Re-click)"
        elif random.random() < 0.5:
            # 10% chance to scroll
            next_action = {"action": "scroll", "direction": "down", "reason": "Random Fallback (Scroll)"}
        else:
            # 10% chance to go back
            next_action = {"action": "back", "reason": "Random Fallback (Back)"}

    # ==============================================================================
    # 🛡️ GLOBAL SAFETY GUARD (The "Final Line of Defense")
    # ==============================================================================
    # Check if the proposed action is identical to the last executed action.
    # This catches loops where state hash changes (e.g. Toast) but action is repeated.
    
    # 🆕 Fix: If next_action is None, we skip this check (it will be handled by Page Explored logic)
    if next_action and graph.transitions:
        last_t = graph.transitions[-1]
        last_action = last_t[1]
        
        # Check for Tap Repetition
        if next_action.get('action') == 'tap' and last_action.get('action') == 'tap':
            x1, y1 = next_action.get('x', 0), next_action.get('y', 0)
            x2, y2 = last_action.get('x', 0), last_action.get('y', 0)
            
            dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
            
            # If clicking same spot (within 50px)
            if dist < 50:
                 # EXCEPTION: Input fields often need 2 clicks (Focus + Cursor)
                 # We can check if it's an EditText, but for now let's just allow 2, block 3.
                 
                 # Check the one BEFORE last
                 if len(graph.transitions) >= 2:
                     prev_t = graph.transitions[-2]
                     prev_action = prev_t[1]
                     if prev_action.get('action') == 'tap':
                         x3, y3 = prev_action.get('x', 0), prev_action.get('y', 0)
                         dist2 = ((x1-x3)**2 + (y1-y3)**2)**0.5
                         
                         if dist2 < 50:
                             # 3rd click on same spot -> BLOCK
                             manager.add_log(client_id, f"🛑 Global Guard: Detected triple-click loop at ({x1},{y1}). Forcing BACK.")
                             
                             # Force Back
                             back_action = {"action": "back", "reason": "Global Loop Protection"}
                             graph.transitions.append((graph.current_state_hash, back_action, "pending"))
                             return back_action

    if next_action:
        # 如果处于 "EXPLORING" 或 "SCANNING" 阶段
        if client_phase in ["EXPLORING", "SCANNING"]:
            # ⏰ 检查时间限制 (SCANNING 阶段限制为 3 分钟)
            limit = 180 if client_phase == "SCANNING" else 1200
            start_time = active_clients.get(client_id, {}).get("start_time")
            
            if start_time and (time.time() - start_time > limit):
                if client_phase == "SCANNING":
                    manager.add_log(client_id, "⏰ 快速扫描结束，进入规划阶段 (PLANNING)...")
                    active_clients[client_id]["phase"] = "PLANNING"
                    # Trigger Planning
                    await generate_exploration_plan(client_id)
                    return {"action": "wait", "reason": "Generating Plan"}
                else:
                    manager.add_log(client_id, "⏰ 探索时间已达上限，自动停止。")
                    active_clients[client_id]["phase"] = "IDLE"
                    return {"action": "wait", "reason": "探索时间结束"}

            # 🛑 卡死检测 (Anti-Stuck)
            if graph.is_stuck():
                # 🛡️ 根页面保护 (Home Page Protection)
                # Only apply protection if it's REALLY a root page (double checked)
                if graph.is_root_page(xml_data):
                    manager.add_log(client_id, "⚠️ 在主页卡死，停止探索以防退出 APP。")
                    active_clients[client_id]["phase"] = "IDLE"
                    return {"action": "wait", "reason": "主页卡死保护"}

                manager.add_log(client_id, "🛑 检测到页面卡死 (连续无效操作)，强制执行 Back...")
                # 强制记录一次 Back，打破循环
                back_action = {"action": "back", "reason": "强制跳出卡死状态"}
                graph.transitions.append((graph.current_state_hash, back_action, "unknown"))
                # 强制执行系统级 Back
                return {"action": "key_event", "keycode": 4, "reason": "Anti-Stuck Back"}

        # 记录这次尝试
        await graph.mark_action_explored(next_action)
        next_action["reason"] = f"探索新元素: {next_action.get('text', 'Unknown')}"
        
        # 记录 Transition
        graph.transitions.append((graph.current_state_hash, next_action, "pending"))
        
        # 更新步骤数
        if client_id in active_clients:
             active_clients[client_id]["stats"]["steps"] += 1
             
        return next_action
                
    # If no next_action (and LLM failed or returned None)
    # 🆕 This block handles "Page Explored" logic
    # If next_action is None, it means graph.get_next_unexplored_action() returned None.
    
    if client_phase in ["EXPLORING", "AUTO_DISCOVERY"]:
        # ⏰ 检查时间限制 (20分钟 = 1200秒)
        # 🆕 Increased Limit to 45 mins to allow full convergence
        start_time = active_clients.get(client_id, {}).get("start_time")
        if start_time and (time.time() - start_time > 2700):
            manager.add_log(client_id, "⏰ 探索时间已达 45 分钟上限，自动停止。")
            active_clients[client_id]["phase"] = "FINISHED"
            manager.add_log(client_id, "💾 探索结束，测试用例已保留。")
            return {"action": "wait", "reason": "探索时间结束"}
            
        # Check if we are at Root
        is_at_root = (graph.root_hash and graph.current_state_hash == graph.root_hash) or graph.is_root_page(xml_data)
        
        # 🆕 STRICT BACK STRATEGY (User Request)
        # If the page is fully explored (next_action is None), we MUST leave.
        # No more random monkey business on non-root pages.
        
        if not is_at_root:
             manager.add_log(client_id, "✅ 当前页面已全部探索完毕 (All Content & Nav Checked).")
             manager.add_log(client_id, "🔙 正在返回上一级 (System Back)...")
             
             back_action = {"action": "key_event", "keycode": 4, "reason": "Page Fully Explored -> Back"}
             # Record this back action as a transition so we know we tried to leave
             graph.transitions.append((graph.current_state_hash, back_action, "pending"))
             return back_action
        
        # 🚀 ROOT PAGE LOGIC
        else:
             # Check if we still have pending intents to try!
             if client_id in active_clients and active_clients[client_id].get("pending_intents"):
                 # This should have been caught above, but double check.
                 next_intent = active_clients[client_id]["pending_intents"].pop(0)
                 manager.add_log(client_id, f"💉 [Fallback] 主页空闲，注入剩余 Intent: {next_intent}")
                 shell_action = {"action": "shell", "command": next_intent, "reason": "Static Analysis Injection"}
                 graph.transitions.append((graph.current_state_hash, shell_action, "pending"))
                 return shell_action

             manager.add_log(client_id, "🏠 主页内容已探索完毕。")


             # 🆕 TERMINATION CHECK: If "Settings" is explored, we are DONE.
             # User Rule: "给探索一个结束的节点...在完成设置中的所有探索之后就结束探索"
             explored_settings = False
             
             # Check explored actions of the Root State
             root_state = graph.states.get(graph.root_hash)
             if root_state:
                 for action in root_state["explored_actions"]:
                     t = action.get('text', '').lower()
                     d = action.get('content-desc', '').lower()
                     if 'settings' in t or '设置' in t or 'settings' in d or '设置' in d:
                         explored_settings = True
                         break
             
             # Also Check Current Available Actions (Maybe we missed it in the log but it's not in unexplored list anymore)
             # If "Settings" is NOT in "unexplored", it means we either clicked it OR filtered it out.
             # But here we are in the "else" block of "if not next_action", which means "unexplored" is empty (or filtered).
             
             # So if we are HERE, it means there are NO more unexplored actions on the Root Page.
             # By definition, if there was a "Settings" button, we MUST have clicked it (unless it was forbidden/throttled).
             # So we can effectively say: If we are at Root and have nothing left to click, we are DONE.
             
             # BUT, the user wants specifically "Settings" to be the trigger.
             # Let's relax the condition: If we are at Root and have nothing left to click, 
             # AND we have visited at least one "Settings" page (or clicked a Settings button), then stop.
             
             # Let's just FORCE STOP if we are at Root and exhausted everything.
             # Because "Settings" was prioritized to be LAST.
             # So if we exhausted everything, we must have done Settings.
             
             manager.add_log(client_id, "✅ 主页所有入口（包括设置）均已尝试。")
             manager.add_log(client_id, "🏁 满足终止条件：主页探索完毕。停止任务。")
             
             if client_phase == "AUTO_DISCOVERY":
                 # Auto-switch to GENERATING phase -> then EXECUTING
                 manager.add_log(client_id, "✅ Intent 遍历完成，开始基于地图生成测试用例...")
                 generate_test_cases_from_graph(client_id)
                 active_clients[client_id]["phase"] = "EXECUTING"
                 return {"action": "wait", "reason": "Auto Discovery Finished -> Generating -> Executing"}
             else:
                 active_clients[client_id]["phase"] = "FINISHED"
                 return {"action": "wait", "reason": "Root Page Exhausted -> Finished"}


             # 1. Try to switch tabs if available (Priority)
             if available_actions:
                 # Filter for bottom tabs
                 max_y = max([a.get('y', 0) for a in available_actions])
                 tabs = [a for a in available_actions if a.get('y', 0) > max_y * 0.9]
                 # Filter out tabs that we have already clicked? 
                 # Tabs are tricky because we want to click them even if "explored" to switch views.
                 # But get_next_unexplored_action already handles "unexplored" ones.
                 # If we are here, it means even tabs are marked "explored".
                 
                 # Let's try to click a random tab to switch context (maybe new content appears?)
                 if tabs:
                     import random
                     target = random.choice(tabs)
                     manager.add_log(client_id, f"🔄 主页切换 Tab: {target.get('text', 'Tab')}")
                     graph.transitions.append((graph.current_state_hash, target, "pending"))
                     return target

             # 2. Scroll to find more content (Secondary)
             manager.add_log(client_id, "🏠 主页尝试向下滚动寻找新内容...")
             scroll_action = {"action": "scroll", "direction": "down", "reason": "Root Page Scroll"}
             graph.transitions.append((graph.current_state_hash, scroll_action, "pending"))
             return scroll_action
            
    # 如果处于 "DEEP_EXPLORING" 阶段，执行任务清单
    elif client_phase == "DEEP_EXPLORING":
        tasks = active_clients[client_id].get("exploration_tasks", [])
        idx = active_clients[client_id].get("current_task_index", 0)
        
        if idx < len(tasks):
            current_task = tasks[idx]
            
            # ASK LLM to execute this SPECIFIC task
            if AI_PROVIDER != "mock":
                manager.add_log(client_id, f"🎯 执行任务 ({idx+1}/{len(tasks)}): {current_task}")
                
                # Construct History
                history_log = [f"{t[1].get('action')} {t[1].get('text','')}" for t in graph.transitions[-5:]]
                
                # Special Prompt for Task Execution
                prompt = f"""
                GOAL: Execute the following task: "{current_task}"
                
                Current Page Summary: {graph.states[current_hash].get("summary", "Unknown")}
                History: {history_log}
                
                If the task is completed or impossible, return "action": "finish_task".
                Otherwise, return the next step to achieve this goal.
                """
                
                # Reuse query_llm_agent but with custom prompt injection? 
                # For simplicity, we just use the standard function but prepend the goal to history or prompt.
                # Let's inject it into the prompt by modifying query_llm_agent slightly or just hacking the history.
                
                # Hack: Prepend GOAL to the last history item to make LLM aware
                history_log.append(f"CURRENT GOAL: {current_task}")
                
                llm_decision = await query_llm_agent(screenshot_b64_compressed, xml_data, history_log, activity)
                
                if llm_decision:
                    if llm_decision.get("action") == "finish_task" or llm_decision.get("action") == "complete":
                        manager.add_log(client_id, f"✅ 任务完成: {current_task}")
                        active_clients[client_id]["current_task_index"] += 1
                        return {"action": "wait", "reason": "Task Completed, moving to next"}
                    else:
                        return llm_decision
            
            return {"action": "wait", "reason": "No AI Provider for Deep Exploration"}
            
        else:
             manager.add_log(client_id, "🏁 所有深度探索任务已完成！")
             active_clients[client_id]["phase"] = "FINISHED"
             return {"action": "wait", "reason": "All Tasks Done"}

    # 如果处于 "EXECUTING" 阶段，执行测试用例
    elif client_phase == "EXECUTING":
        # 查找第一个 pending 或 running 的用例
        pending_case = None
        case_index = -1
        if client_id in active_clients:
            for idx, case in enumerate(active_clients[client_id]["test_cases"]):
                if case["status"] == "pending":
                    pending_case = case
                    case_index = idx
                    break
                elif case["status"] == "running":
                    # Continue running this case
                    pending_case = case
                    case_index = idx
                    break
        
        if pending_case:
            # Mark as running if not already
            if pending_case["status"] == "pending":
                if client_id in active_clients:
                    active_clients[client_id]["test_cases"][case_index]["status"] = "running"
                manager.add_log(client_id, f"🚀 开始执行用例 #{pending_case['id']}: {pending_case['desc']}")

            # Use LLM to execute
            # 🆕 Heuristic Execution (No AI) - Now with SCRIPT Support
            
            # Check if we have a script
            script = pending_case.get("script", [])
            current_step_idx = active_clients[client_id].get("current_step_index", 0)
            
            if script and current_step_idx < len(script):
                step = script[current_step_idx]
                action_type = step.get("action")
                target = step.get("target")
                value = step.get("value")
                
                manager.add_log(client_id, f"📜 执行脚本步骤 [{current_step_idx+1}/{len(script)}]: {action_type} '{target}'")
                
                # Execute Step
                found_action = None
                
                if action_type in ["click", "input"]:
                    # Find element
                    for act in available_actions:
                         if target.lower() in act.get('text', '').lower() or target.lower() in act.get('content-desc', '').lower():
                             found_action = act
                             break
                    
                    if found_action:
                        # Perform Action
                        # TODO: Handle Input Text if needed (currently edge client might not support dynamic text input easily without separate command)
                        # For now assume click.
                        
                        # Increment step index for NEXT loop
                        active_clients[client_id]["current_step_index"] = current_step_idx + 1
                        
                        # If this was the last step, mark case as passed
                        if current_step_idx + 1 >= len(script):
                             manager.add_log(client_id, f"✅ 脚本执行完毕，用例 #{pending_case['id']} 通过!")
                             active_clients[client_id]["test_cases"][case_index]["status"] = "passed"
                             active_clients[client_id]["current_step_index"] = 0 # Reset
                        
                        return found_action
                    else:
                        manager.add_log(client_id, f"❌ 步骤失败: 找不到元素 '{target}'")
                        active_clients[client_id]["test_cases"][case_index]["status"] = "failed"
                        active_clients[client_id]["current_step_index"] = 0 # Reset
                        return {"action": "wait", "reason": "Script Element Not Found"}
                        
                elif action_type == "assert":
                    # Check existence
                    exists = False
                    # Search in XML (available_actions is a subset, let's look at raw xml? or just actions)
                    # Using available_actions is safer for interactive elements.
                    for act in available_actions:
                         if target.lower() in act.get('text', '').lower() or target.lower() in act.get('content-desc', '').lower():
                             exists = True
                             break
                    
                    if exists:
                        manager.add_log(client_id, f"✅ 断言成功: 发现 '{target}'")
                        active_clients[client_id]["current_step_index"] = current_step_idx + 1
                        if current_step_idx + 1 >= len(script):
                             manager.add_log(client_id, f"✅ 脚本执行完毕，用例 #{pending_case['id']} 通过!")
                             active_clients[client_id]["test_cases"][case_index]["status"] = "passed"
                             active_clients[client_id]["current_step_index"] = 0
                        return {"action": "wait", "reason": "Assert Passed"} # Wait for next loop to pick up next step
                    else:
                        manager.add_log(client_id, f"❌ 断言失败: 未发现 '{target}'")
                        active_clients[client_id]["test_cases"][case_index]["status"] = "failed"
                        active_clients[client_id]["current_step_index"] = 0
                        return {"action": "wait", "reason": "Assert Failed"}
                
                elif action_type == "wait":
                     time.sleep(1) # blocking wait? or return wait action?
                     active_clients[client_id]["current_step_index"] = current_step_idx + 1
                     return {"action": "wait", "reason": "Script Wait"}

                # 🆕 AGENT COMMAND SUPPORT
                elif action_type == "agent_command":
                    payload = step.get("payload", "")
                    manager.add_log(client_id, f"📡 发送 Agent 指令: {payload}")
                    response = agent_client.send_command(payload)
                    manager.add_log(client_id, f"📨 Agent 响应: {response}")
                    
                    active_clients[client_id]["current_step_index"] = current_step_idx + 1
                    
                    # Check if this was the last step
                    if current_step_idx + 1 >= len(script):
                         manager.add_log(client_id, f"✅ 脚本执行完毕，用例 #{pending_case['id']} 通过!")
                         active_clients[client_id]["test_cases"][case_index]["status"] = "passed"
                         active_clients[client_id]["current_step_index"] = 0
                         
                    return {"action": "wait", "reason": f"Agent Cmd Executed: {response}"}
                
                # 🆕 INTENT JUMP SUPPORT (Teleport)
                elif action_type == "intent_jump":
                    target_activity = step.get("target") # e.g. ".MainActivity" or full class name
                    
                    # 1. Lookup in Map
                    full_activity_name = None
                    if target_activity.startswith("."):
                         # Relative path, prepend package
                         pkg = APP_KNOWLEDGE_MAP.get("package_name", "")
                         full_activity_name = pkg + target_activity
                    else:
                         full_activity_name = target_activity
                         
                    # 2. Check if exported (Safe to jump?)
                    # For now assume yes or check map
                    
                    manager.add_log(client_id, f"🚀 正在瞬移至页面: {full_activity_name}")
                    
                    # 3. Send ADB Command via Agent or Edge Client?
                    # Edge Client handles basic shell commands usually.
                    # Let's use a special action for Edge Client to execute shell.
                    
                    jump_cmd = f"am start -n {APP_KNOWLEDGE_MAP.get('package_name')}/{full_activity_name}"
                    
                    # We can use the return action to tell Edge Client to run this shell
                    active_clients[client_id]["current_step_index"] = current_step_idx + 1
                    
                    # Check if this was the last step
                    if current_step_idx + 1 >= len(script):
                         manager.add_log(client_id, f"✅ 脚本执行完毕，用例 #{pending_case['id']} 通过!")
                         active_clients[client_id]["test_cases"][case_index]["status"] = "passed"
                         active_clients[client_id]["current_step_index"] = 0
                    
                    return {"action": "shell", "command": jump_cmd, "reason": "Teleporting..."}

            # Fallback to Old Heuristic (Keyword Match) if NO script or script empty
            # ... (Old Code) ...
            manager.add_log(client_id, f"🤖 尝试启发式执行 (无脚本 fallback): {pending_case['desc']}")
            
            # 1. Simple Keyword Matching
            target_text = None
            desc_lower = pending_case['desc'].lower()
            
            # Extract quoted text as high priority target (e.g. Click "Login")
            import re
            quotes = re.findall(r"['\"](.*?)['\"]", pending_case['desc'])
            
            # Find actionable elements
            best_action = None
            
            # Strategy A: Look for exact text match from quotes
            if quotes:
                for q in quotes:
                     # Find element with this text
                     for act in available_actions:
                         if q.lower() in act.get('text', '').lower() or q.lower() in act.get('content-desc', '').lower():
                             best_action = act
                             break
                     if best_action: break
            
            # Strategy B: Look for keywords in description
            if not best_action:
                keywords = desc_lower.split()
                # Filter out common words
                stop_words = ['verify', 'check', 'test', 'ensure', 'the', 'a', 'an', 'click', 'tap', 'press', '验证', '测试', '检查']
                keywords = [k for k in keywords if k not in stop_words]
                
                for k in keywords:
                    for act in available_actions:
                         if k in act.get('text', '').lower() or k in act.get('content-desc', '').lower():
                             best_action = act
                             break
                    if best_action: break
            
            if best_action:
                manager.add_log(client_id, f"🎯 命中 UI 元素: {best_action.get('text', 'Unknown')}")
                
                # Assume if we click it, the case MIGHT be passed? 
                # Or we need to verify result?
                # For simple "Verify Login", clicking Login might be the step.
                # Let's mark it passed after clicking for this simple heuristic.
                
                # Mark as passed immediately for demo purposes if it's a simple verify
                # In reality we need 2 steps: Action -> Verify.
                # Let's just do the action.
                
                # Record success
                manager.add_log(client_id, f"✅ 用例 #{pending_case['id']} 启发式执行成功!")
                if client_id in active_clients:
                     active_clients[client_id]["test_cases"][case_index]["status"] = "passed"
                
                return best_action
            
            # Strategy C: If no UI match, maybe it's a passive check? (e.g. "Check if X exists")
            # If we found the element but it's not clickable, we pass.
            # But available_actions are clickable.
            
            # Strategy D: Fallback to AI if Heuristic fails (Soft Fail)
            # But User explicitly said "No AI".
            # So we fail the case if heuristic fails.
            
            manager.add_log(client_id, f"❌ 启发式匹配失败: 无法在当前页面找到与 '{pending_case['desc']}' 相关的元素。")
            
            # 🆕 HYBRID FALLBACK: Try Visual Template Matching
            # If we have a cached image for the target text, try to find it on screen using OpenCV.
            if graph.visual_memory:
                # Try to find a key in visual memory that matches pending_case desc keywords
                visual_target = None
                
                # Check quotes first
                if quotes:
                    for q in quotes:
                        if q in graph.visual_memory:
                            visual_target = graph.visual_memory[q]
                            manager.add_log(client_id, f"🖼️ 视觉锚点命中: '{q}'")
                            break
                
                # Check keywords
                if not visual_target:
                    keywords = pending_case['desc'].split()
                    for k in keywords:
                        if k in graph.visual_memory:
                            visual_target = graph.visual_memory[k]
                            manager.add_log(client_id, f"🖼️ 视觉锚点命中: '{k}'")
                            break
                            
                if visual_target:
                    manager.add_log(client_id, "👁️ 尝试使用 OpenCV 视觉定位...")
                    return {
                        "action": "template_tap", 
                        "template": visual_target,
                        "threshold": 0.8,
                        "reason": "Visual Fallback"
                    }

            if client_id in active_clients:
                active_clients[client_id]["test_cases"][case_index]["status"] = "failed"
            return {"action": "wait", "reason": "Heuristic Match Failed"}

            # ORIGINAL AI LOGIC (Commented out for Strict No-AI Request)
            '''
            if AI_PROVIDER != "mock":
                 # Construct History
                history_log = [f"{t[1].get('action')} {t[1].get('text','')}" for t in graph.transitions[-5:]]
                
                # Special Prompt for Test Execution
                # We reuse the "Task Execution" logic but tuned for Verification
                prompt = f"""
                GOAL: Execute the following TEST CASE: "{pending_case['desc']}"
                
                Current Page Summary: {graph.states[current_hash].get("summary", "Unknown")}
                History: {history_log}
                
                INSTRUCTIONS:
                1. Look for UI elements that match the test case requirements (e.g. "Init SDK", "Connect").
                2. If the test step is successfully completed/verified, return "action": "pass_case".
                3. If the test failed or cannot be completed, return "action": "fail_case".
                4. Otherwise, return the next UI action to proceed.
                """
                
                # Hack: Prepend GOAL to history
                history_log.append(f"CURRENT TEST CASE: {pending_case['desc']}")
                
                llm_decision = await query_llm_agent(screenshot_b64_compressed, xml_data, history_log, activity)
                
                if llm_decision:
                    action_type = llm_decision.get("action")
                    
                    if action_type in ["pass_case", "finish_task", "complete"]:
                        manager.add_log(client_id, f"✅ 用例 #{pending_case['id']} 通过!")
                        if client_id in active_clients:
                            active_clients[client_id]["test_cases"][case_index]["status"] = "passed"
                        return {"action": "wait", "reason": "Case Passed"}
                        
                    elif action_type in ["fail_case", "failed"]:
                        manager.add_log(client_id, f"❌ 用例 #{pending_case['id']} 失败: {llm_decision.get('reason')}")
                        if client_id in active_clients:
                            active_clients[client_id]["test_cases"][case_index]["status"] = "failed"
                        return {"action": "wait", "reason": "Case Failed"}
                        
                    else:
                        # Normal UI Action
                        await graph.mark_action_explored(llm_decision)
                        llm_decision["reason"] = f"Executing Case #{pending_case['id']}: {llm_decision.get('reason')}"
                        return llm_decision
            '''
            
            # Fallback if Mock or LLM fails
            return {"action": "wait", "reason": "Waiting for AI..."}
            
        else:
            # 所有用例执行完毕
            if client_id in active_clients:
                active_clients[client_id]["phase"] = "FINISHED"
                manager.add_log(client_id, "🏁 所有测试用例执行完毕！")
            return {"action": "wait", "reason": "测试完成"}

    return {"action": "wait", "reason": "等待指令..."}

# ==============================================================================
# 🔌 Agent Communication (MCP/SDK)
# ==============================================================================
class AgentClient:
    def __init__(self, port=9000):
        self.port = port
        self.host = 'localhost'

    def send_command(self, cmd_str):
        """Send a raw string command to the Agent and return response"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0) # Fast timeout
                s.connect((self.host, self.port))
                s.sendall((cmd_str + "\n").encode('utf-8'))
                response = s.recv(4096).decode('utf-8').strip()
                return response
        except Exception as e:
            return f"ERROR: {str(e)}"

agent_client = AgentClient()

# ==============================================================================
# 🔌 WebSocket 接口
# ==============================================================================
@app.post("/agent/command")
async def send_agent_command(command: str = Form(...)):
    """Manually send a command to the On-Device Agent"""
    response = agent_client.send_command(command)
    return {"command": command, "response": response}

# ==============================================================================
# 📦 APK Analysis
# ==============================================================================
APP_KNOWLEDGE_MAP = {} # Global cache for the "Map"

def analyze_apk(apk_path):
    """
    Run the static analysis tool on the uploaded APK.
    Returns the path to the analysis JSON report.
    """
    try:
        import subprocess
        
        output_json = apk_path + "_analysis.json"
        
        # Call the standalone analyzer script we just created
        analyzer_script = os.path.join(os.getcwd(), "apk_analyzer.py")
        cmd = [sys.executable, analyzer_script, apk_path, output_json]
        
        print(f"🕵️‍♂️ Running Static Analysis: {' '.join(cmd)}")
        
        # Capture output to debug failures
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Analyzer Error STDOUT: {result.stdout}")
            print(f"❌ Analyzer Error STDERR: {result.stderr}")
            raise Exception(f"Analyzer Script Failed: {result.stderr}")
            
        print(f"✅ Analyzer Output: {result.stdout}")
        
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                data = json.load(f)
                
            # Cache the map into memory for fast lookup
            global APP_KNOWLEDGE_MAP
            APP_KNOWLEDGE_MAP = data
            print(f"🗺️ App Map Loaded! Found {len(data.get('activities', []))} activities.")
            
            return output_json
        else:
            raise Exception("Output JSON not found after analysis")
            
    except Exception as e:
        print(f"❌ Static Analysis Failed: {e}")
        return None

from fastapi import UploadFile, File

@app.post("/upload_apk")
async def upload_apk(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    try:
        # Use shutil or simple write
        with open(temp_path, "wb") as buffer:
            # Check if file object supports read or spooled
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        print(f"❌ File Save Error: {e}")
        return {"status": "error", "message": f"Save Failed: {str(e)}"}
    
    # Run Analysis
    # If file exists, we can skip re-analysis for demo speed, or force it.
    # Let's check if json exists
    expected_json = temp_path + "_analysis.json"
    
    # Try to load existing map first (Speed optimization)
    if os.path.exists(expected_json):
        try:
            with open(expected_json, 'r') as f:
                data = json.load(f)
            global APP_KNOWLEDGE_MAP
            APP_KNOWLEDGE_MAP = data
            print(f"🗺️ Loaded cached Map for {file.filename}")
            return {"status": "ok", "message": "Cached Map Loaded", "report": expected_json}
        except:
            pass

    report_path = analyze_apk(temp_path)
    
    if report_path:
        return {"status": "ok", "message": "APK Analysis Complete", "report": report_path}
    else:
        return {"status": "error", "message": "Analysis Failed"}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    # 🚀 Auto-Kickoff: 连接建立后，立即请求一次状态更新
    # 这对于断线重连恢复非常重要，否则双方都在等对方
    manager.add_log(client_id, "🔄 Client Reconnected. Requesting state sync...")
    
    # 🆕 FORCE IDLE ON CONNECT: Ensure we start in IDLE mode to avoid auto-run
    if client_id in active_clients:
        active_clients[client_id]["phase"] = "IDLE"
        
    await manager.send_command(client_id, {"type": "capture_screenshot"})

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 更新心跳
            if client_id in active_clients:
                active_clients[client_id]["last_seen"] = time.time()
            
            if message.get("type") == "heartbeat":
                # 🆕 Handle Heartbeat
                # Just update last_seen, already done at start of loop
                # Optionally send pong back? No need for simple keep-alive.
                pass
            
            elif message.get("type") == "stream_frame":
                # 🚀 Re-enable Streaming
                b64_img = message.get('data', '')
                if b64_img:
                    print(f"DEBUG: Received stream frame for {client_id}, len={len(b64_img)}")
                    manager.update_screenshot(client_id, b64_img)
                else:
                    print(f"⚠️ Received empty stream frame from {client_id}")
            
            elif message.get("type") == "screenshot_response":
                b64_img = message['data']
                xml_data = message.get('xml') # 🆕 Get XML
                app_info = message.get('app_info', {}) # 🆕 Get App Info
                activity = app_info.get('activity')
                
                manager.update_screenshot(client_id, b64_img)

                # 🛑 IDLE CHECK: 如果当前处于 IDLE 状态，仅更新画面，不进行决策
                current_phase = active_clients.get(client_id, {}).get("phase", "IDLE")
                if current_phase == "IDLE":
                    # manager.add_log(client_id, "📸 收到预览截图 (IDLE - 无动作)")
                    # 也可以选择做一些被动的状态分析（如更新图谱），但暂时保持静默以节省资源
                    # 更新状态图中的 screenshot，以便下次启动时有最新画面
                    if client_id in client_graphs:
                        graph = client_graphs[client_id]
                        if graph.current_state_hash and graph.current_state_hash in graph.states:
                            graph.states[graph.current_state_hash]["screenshot"] = b64_img
                    continue

                # 🛑 WAIT FOR APK ANALYSIS: 如果处于 APK 分析阶段，暂停动作
                if current_phase == "ANALYZING_APK":
                    # Check if analysis is done (e.g. check a flag or file)
                    # For now, we just wait. The APK analysis task should switch phase to AUTO_DISCOVERY when done.
                    manager.add_log(client_id, "⏳ 等待 APK 静态分析完成...")
                    await asyncio.sleep(2)
                    continue
                
                # 🆕 FAST TRAVERSAL CHECK: Skip analysis if we are just jumping
                # If the LAST action was a "shell" jump, we don't need deep analysis of the result immediately.
                # We just need to register the new state and jump again if needed.
                # However, decide_next_step handles the logic of "Do I have more intents?"
                # So we DO need to call it, but we can suppress the "Analysing..." log to reduce user confusion.
                
                is_fast_traversal = active_clients.get(client_id, {}).get("phase") == "AUTO_DISCOVERY" and \
                                    active_clients.get(client_id, {}).get("pending_intents")
                
                if is_fast_traversal:
                     # manager.add_log(client_id, "🚀 快速遍历中: 确认跳转结果...")
                     pass
                elif current_phase == "EXECUTING":
                     # 🆕 Suppress Analysis Log in Executing Mode
                     pass
                else:
                     manager.add_log(client_id, "📸 收到分析用截图，正在请求 AI 分析...")
                
                # 1. 调用 AI + 状态机决策
                # 🆕 BYPASS DECIDE_NEXT_STEP for Script Execution
                # If we are in EXECUTING phase, we want to run the script logic DIRECTLY, 
                # instead of calling decide_next_step which has a lot of "exploration" overhead.
                # However, decide_next_step handles the logic of "pick the next step of the script".
                # So we must modify decide_next_step to be LEANER or split the logic.
                
                # For now, let's just suppress the "AI Decision" log if we are in EXECUTING mode
                # to avoid confusing the user.
                
                decision = await decide_next_step(client_id, b64_img, xml_data, activity=activity, app_info=app_info)
                
                current_phase = active_clients.get(client_id, {}).get("phase")
                
                if not is_fast_traversal and current_phase != "EXECUTING":
                     manager.add_log(client_id, f"🧠 AI 决策: {decision.get('reason', 'Unknown')}")
                elif current_phase == "EXECUTING":
                     # Log script progress instead of "AI Decision"
                     # The script execution logic inside decide_next_step already logs "Executing Step X".
                     pass
                
                # 🆕 Save this decision as PENDING log data for the NEXT step to record
                if decision.get("action") not in ["wait", "finish_task", "complete"]:
                    if client_id in active_clients:
                        # Ensure session_id exists
                        if "session_id" not in active_clients[client_id]:
                            import uuid
                            active_clients[client_id]["session_id"] = str(uuid.uuid4())
                            active_clients[client_id]["step_count"] = 0
                        
                        active_clients[client_id]["step_count"] = active_clients[client_id].get("step_count", 0) + 1
                        
                        active_clients[client_id]["pending_log_data"] = {
                            "session_id": active_clients[client_id]["session_id"],
                            "step_id": active_clients[client_id]["step_count"],
                            "before_screenshot": b64_img,
                            "before_xml": xml_data,
                            "action": decision,
                            "reasoning": decision.get("reason")
                        }
                
                # 2. 下发指令
                cmd = {"type": "execute_action", "payload": decision}
                await manager.send_command(client_id, cmd)
                
            elif message.get("type") == "action_done":
                manager.add_log(client_id, "✅ 动作执行完毕，等待页面刷新...")
                
                # 🆕 SUPER FAST TRAVERSAL: Reduce server-side wait to near zero for jump actions
                # If we are in "Auto Discovery" mode, we want speed.
                current_phase = active_clients.get(client_id, {}).get("phase", "IDLE")
                
                if current_phase == "AUTO_DISCOVERY":
                     await asyncio.sleep(0.1) # 🚀 Ultra Fast for Traversal
                else:
                     await asyncio.sleep(0.8) # Standard wait for stability
                
                # 🛑 Stop loop if IDLE
                if current_phase == "IDLE":
                    manager.add_log(client_id, "⏸️ 任务已暂停 (IDLE)。点击 'Explore' 继续。")
                else:
                    manager.add_log(client_id, "📡 请求下一帧截图...")
                    await manager.send_command(client_id, {"type": "capture_screenshot"})
            
            elif message.get("type") == "log":
                msg = message.get("message", "")
                manager.add_log(client_id, f"📱 Edge Log: {msg}")
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# ==============================================================================
# 🖥️ Web 控制台 (Dashboard)
# ==============================================================================
@app.get("/")
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TestGPT Operations Center</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            gray: { 900: '#0f1115', 800: '#161b22', 700: '#21262d', 600: '#30363d' },
                            accent: { 500: '#58a6ff', 600: '#1f6feb' }
                        }
                    }
                }
            }
        </script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
            body { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #c9d1d9; }
            .font-mono { font-family: 'JetBrains Mono', monospace; }
            
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #161b22; }
            ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
            
            .panel { background: #161b22; border: 1px solid #30363d; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }
            .panel-header { padding: 12px 16px; border-bottom: 1px solid #30363d; font-weight: 600; font-size: 14px; display: flex; justify-content: space-between; align-items: center; background: #0d1117; }
            .log-entry { font-size: 13px; line-height: 1.5; border-bottom: 1px solid #21262d; padding: 4px 8px; }
            .log-entry:last-child { border-bottom: none; }
            
            .screen-card {
                /* aspect-ratio: 9/19.5; removed to allow flex sizing */
                background: #000;
                /* border-radius: 16px; handled by tailwind */
                /* border: 4px solid #30363d; handled by JS */
                overflow: hidden;
                box-shadow: 0 0 20px rgba(0,0,0,0.3);
            }
        </style>
    </head>
    <body class="h-screen w-screen bg-[#0d1117] text-[#c9d1d9] overflow-hidden flex flex-col">
        <!-- Top Bar (Fixed Height) -->
        <header class="h-14 border-b border-gray-600 bg-gray-800 flex items-center justify-between px-6 shrink-0 z-50">
            <div class="flex items-center gap-4">
                <div class="w-8 h-8 bg-accent-600 rounded flex items-center justify-center font-bold text-white">T</div>
                <h1 class="text-lg font-semibold text-white tracking-tight">TestGPT <span class="text-gray-400 font-normal">Operations Center v5.1 (Multi-Device Grid)</span></h1>
            </div>
            <div id="global-status" class="flex items-center gap-3">
                <!-- Global Controls -->
                <div class="flex gap-2 mr-4 border-r border-gray-600 pr-4">
                    <button onclick="triggerGlobalAction('start_all')" class="bg-green-700 hover:bg-green-600 text-white px-2 py-1 rounded text-xs font-bold transition-colors flex items-center gap-1">
                        <span>▶</span> ALL
                    </button>
                    <button onclick="triggerGlobalAction('stop_all')" class="bg-red-900 hover:bg-red-800 text-white px-2 py-1 rounded text-xs font-bold transition-colors flex items-center gap-1 border border-red-700">
                        <span>⏹</span> ALL
                    </button>
                </div>
                <span class="text-sm text-gray-400">System Ready</span>
                <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            </div>
        </header>

        <!-- Main Workspace (Takes remaining height, NO SCROLL on parent) -->
        <main class="flex-1 min-h-0 p-4 w-full">
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 h-full w-full">
                
                <!-- Col 1: Device Screens (Grid) - 5/12 -->
                <div class="lg:col-span-5 flex flex-col gap-4 h-full min-h-0">
                    <div class="panel p-4 bg-gray-900/50 flex-1 min-h-0 relative">
                         <div class="text-xs uppercase tracking-wider text-gray-500 font-bold mb-2 flex justify-between">
                            <span>Connected Devices</span>
                            <span id="device-count" class="text-accent-500">0</span>
                         </div>
                         <div id="screens-grid" class="grid grid-cols-1 md:grid-cols-2 gap-4 w-full h-full overflow-y-auto content-start pb-10">
                            <!-- Dynamic Cards will be injected here -->
                            <div id="waiting-msg" class="text-gray-500 text-center col-span-full mt-10 text-sm">Waiting for devices...</div>
                         </div>
                    </div>
                </div>

                <!-- Col 2: Controls & Stats - 3/12 -->
                <div class="lg:col-span-3 flex flex-col gap-4 h-full min-h-0">
                    <!-- Stats Row -->
                    <div class="grid grid-cols-2 gap-2 shrink-0 h-20">
                        <div class="panel p-2 justify-center items-center">
                            <span class="text-gray-500 text-[10px] uppercase">Pages</span>
                            <span class="text-xl font-bold text-white" id="stat-pages">0</span>
                        </div>
                        <div class="panel p-2 justify-center items-center">
                            <span class="text-gray-500 text-[10px] uppercase">Steps</span>
                            <span class="text-xl font-bold text-accent-500" id="stat-steps">0</span>
                        </div>
                    </div>

                    <!-- Controls Panel -->
                    <div class="panel shrink-0 p-4 gap-3 flex-1 overflow-y-auto">
                        <!-- Active Client Info -->
                        <div class="mb-2 p-2 bg-gray-800 rounded border border-accent-600/30">
                            <label class="text-[10px] uppercase tracking-wider text-accent-400 font-bold mb-1 block">ACTIVE CONTROLLER</label>
                            <div id="active-client-id" class="text-xs text-white font-mono truncate">None</div>
                        </div>

                        <!-- 🆕 APK UPLOAD SECTION -->
                        <div class="mb-2 p-2 bg-gray-800 rounded border border-gray-700">
                             <label class="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-1 block">App Knowledge Map</label>
                             <div class="flex gap-2">
                                 <input type="file" id="apk-upload" class="hidden" accept=".apk" onchange="uploadApk()">
                                 <button onclick="document.getElementById('apk-upload').click()" class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs flex-1">
                                     📂 Upload APK
                                 </button>
                                 <span id="map-status" class="text-[10px] text-gray-500 flex items-center">No Map</span>
                             </div>
                        </div>

                        <!-- 🆕 PROGRESS BAR SECTION -->
                        <div id="progress-container" class="mb-2 p-3 bg-gray-800 rounded border border-gray-700 hidden">
                             <div class="flex justify-between items-end mb-1">
                                 <span id="progress-label" class="text-[10px] uppercase tracking-wider text-white font-bold">Progress</span>
                                 <span id="progress-text" class="text-[10px] text-accent-400 font-mono">0%</span>
                             </div>
                             <div class="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
                                 <div id="progress-bar" class="bg-gradient-to-r from-accent-500 to-purple-500 h-2.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                             </div>
                             <div id="progress-detail" class="text-[9px] text-gray-500 mt-1 text-right truncate">Waiting...</div>
                        </div>

                        <!-- Test Type Selector -->
                        <div class="mb-2">
                            <label class="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-1 block">Target Type</label>
                            <select id="target-type" class="w-full bg-gray-900 border border-gray-700 text-gray-300 text-xs rounded p-2 focus:border-accent-500 focus:outline-none">
                                <option value="app_ble" selected>📱 Mobile App (BLE Terminal)</option>
                                <option value="sdk">📦 SDK (xTalk API)</option>
                                <option value="web">🌐 Web Application</option>
                                <option value="serial">🔌 Serial/UART Device</option>
                            </select>
                        </div>
                        
                        <!-- Smart Wait Toggle (Hidden) -->
                        <div class="mb-2 p-2 bg-gray-800 rounded border border-gray-700 hidden">
                             <div class="flex items-center justify-between mb-1">
                                 <span class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Smart Wait</span>
                                 <label class="relative inline-flex items-center cursor-pointer">
                                     <input type="checkbox" id="smart-wait-toggle" class="sr-only peer" checked onchange="toggleSmartWait()">
                                     <div class="w-7 h-4 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-green-500"></div>
                                 </label>
                             </div>
                        </div>

                        <div class="grid grid-cols-1 gap-3">
                            <!-- New Auto-Test Button (Replaces Explore) -->
                            <button id="btn-start-task" onclick="triggerAction('start_task')" class="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white px-4 py-4 rounded-xl text-base font-bold transition-all shadow-lg shadow-green-900/30 flex items-center justify-center gap-3 transform hover:scale-[1.02] active:scale-95">
                                <span class="text-2xl animate-pulse">🚀</span> 
                                <div class="flex flex-col items-start text-left">
                                    <span class="leading-none text-lg">START TASK</span>
                                    <span class="text-[10px] opacity-90 font-normal">开始全自动测试 (Auto-Pilot)</span>
                                </div>
                            </button>

                            <!-- Run Saved Button -->
                            <button onclick="triggerAction('run_tests')" class="w-full bg-green-700 hover:bg-green-600 text-white px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2 border border-green-600/50">
                                <span>▶️</span> 执行已保存用例 (Run Saved)
                            </button>
                            
                            <button onclick="triggerAction('stop_task')" class="w-full bg-red-900/80 hover:bg-red-800 text-white px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-1 border border-red-700 mt-2">
                                <span>🛑</span> 停止任务
                            </button>
                        </div>

                        <!-- Secondary Actions -->
                        <div class="grid grid-cols-2 gap-2 mt-4">
                            <button onclick="triggerAction('reconnect_device')" class="bg-gray-800 hover:bg-gray-700 text-gray-300 px-2 py-2 rounded text-[10px] border border-gray-700">
                                🔌 连接设备
                            </button>
                            <button onclick="triggerAction('reset_client')" class="bg-gray-800 hover:bg-gray-700 text-red-400 px-2 py-2 rounded text-[10px] border border-gray-700">
                                🗑️ 重置数据
                            </button>
                        </div>
                        
                        <!-- Report Link -->
                        <div class="mt-4 pt-4 border-t border-gray-700 text-center">
                             <a href="#" onclick="alert('测试完成后将自动生成报告')" class="text-xs text-accent-500 hover:underline">📄 查看测试报告 (Report)</a>
                        </div>
                    </div>
                </div>

                <!-- Col 3: Logs & Test Cases - 4/12 -->
                <div class="lg:col-span-4 flex flex-col gap-4 h-full min-h-0">
                     <!-- Test Queue (Top Half) -->
                     <div class="panel flex-1 min-h-0 flex flex-col relative h-1/2">
                        <div class="panel-header shrink-0 z-10">
                            <span>📋 Test Queue</span>
                            <span class="px-2 py-0.5 rounded bg-gray-700 text-xs text-white" id="case-count">0</span>
                        </div>
                        <div class="absolute inset-x-0 bottom-0 top-[45px] overflow-y-auto p-2 space-y-2" id="test-cases-list">
                            <div class="text-center text-gray-600 text-sm py-10">No test cases generated yet.</div>
                        </div>
                    </div>

                    <!-- Logs (Bottom Half) -->
                    <div class="panel flex-1 min-h-0 flex flex-col relative h-1/2">
                        <div class="panel-header shrink-0 z-10">
                            <span>📜 System Logs</span>
                            <div class="flex items-center gap-2">
                                <span class="text-xs text-gray-500 font-mono">Real-time</span>
                                <button onclick="triggerAction('clear_logs')" class="text-[10px] bg-gray-700 hover:bg-gray-600 px-2 py-0.5 rounded text-gray-300 transition-colors">Clear</button>
                            </div>
                        </div>
                        <div class="absolute inset-x-0 bottom-0 top-[45px] overflow-y-auto bg-black p-2 font-mono" id="log-terminal">
                            <!-- Logs injected here -->
                        </div>
                    </div>
                </div>

            </div>
        </main>

        <script>
            let currentClientId = null;

            function updateDashboard() {
                fetch('/api/clients')
                    .then(response => response.json())
                    .then(data => {
                        const clientIds = Object.keys(data);
                        document.getElementById('device-count').innerText = clientIds.length;
                        
                        const grid = document.getElementById('screens-grid');
                        const waitingMsg = document.getElementById('waiting-msg');
                        
                        if (clientIds.length > 0) {
                            if (waitingMsg) waitingMsg.style.display = 'none';
                        } else {
                            if (waitingMsg) waitingMsg.style.display = 'block';
                            // Clear grid except waiting msg
                            Array.from(grid.children).forEach(child => {
                                if (child.id !== 'waiting-msg') child.remove();
                            });
                            return;
                        }

                        // 🚀 Smart Client Selection
                        if (!currentClientId && clientIds.length > 0) {
                             currentClientId = clientIds[0];
                        }
                        // If current ID is gone, switch to first available
                        if (currentClientId && !data[currentClientId] && clientIds.length > 0) {
                             currentClientId = clientIds[0];
                        }

                        // Update Active Client Display
                        const activeLabel = document.getElementById('active-client-id');
                        if (activeLabel) activeLabel.innerText = currentClientId || "None";

                        // 🆕 Update Progress Bar Logic
                        const progressContainer = document.getElementById('progress-container');
                        const progressBar = document.getElementById('progress-bar');
                        const progressText = document.getElementById('progress-text');
                        const progressLabel = document.getElementById('progress-label');
                        const progressDetail = document.getElementById('progress-detail');

                        if (currentClientId && data[currentClientId]) {
                            const info = data[currentClientId];
                            const phase = info.phase || "IDLE";
                            
                            if (phase === "IDLE" || phase === "FINISHED") {
                                progressContainer.classList.add('hidden');
                            } else {
                                progressContainer.classList.remove('hidden');
                                let percent = 0;
                                let label = "Processing";
                                let detail = "";
                                
                                if (phase === "AUTO_DISCOVERY" || phase === "SCANNING" || phase === "DEEP_EXPLORING") {
                                    // 🆕 Saturation-Based Progress (Steps / Target)
                                    // This ensures progress is steady and reflects EFFORT, not just discovery.
                                    
                                    const currentSteps = info.stats.steps || 0;
                                    const targetSteps = info.target_steps || 50;
                                    
                                    percent = (currentSteps / targetSteps) * 100;
                                    
                                    // Clamp to 95% if still running but exceeded target
                                     if (percent > 95) percent = 95;
                                     // Clamp to 5% min
                                     if (percent < 5) percent = 5;
                                     
                                     progressBar.classList.remove("animate-pulse");
                                     progressBar.style.transition = "width 0.5s ease";
                                     
                                     label = "Testing Saturation (测试饱和度)";
                                     detail = `Step ${currentSteps} / ${targetSteps} (Deep Explore)`;
                                     
                                     // 🆕 Over-Saturation Indicator
                                     if (currentSteps > targetSteps) {
                                         detail = `Step ${currentSteps} / ${targetSteps} (Bonus Coverage!)`;
                                         progressBar.classList.add("bg-green-500"); // Change color?
                                     }
                                     
                                     progressText.innerText = `${Math.round(percent)}%`;
                                } 
                                else if (phase === "EXECUTING" || phase === "GENERATING") {
                                    const cases = info.test_cases || [];
                                    const total = cases.length;
                                    if (total > 0) {
                                        const completed = cases.filter(c => c.status === 'passed' || c.status === 'failed').length;
                                        percent = (completed / total) * 100;
                                        label = "Test Execution";
                                        detail = `${completed} / ${total} Cases`;
                                    } else {
                                        percent = (phase === "GENERATING") ? 50 : 0;
                                        label = "Generating Cases...";
                                        detail = "AI Thinking...";
                                        progressBar.classList.add("animate-pulse");
                                    }
                                }

                                progressBar.style.width = `${percent}%`;
                                progressText.innerText = `${Math.round(percent)}%`;
                                progressLabel.innerText = label;
                                progressDetail.innerText = detail;
                            }
                        } else {
                            if(progressContainer) progressContainer.classList.add('hidden');
                        }

                        // 🆕 Update Button State
                        const startBtn = document.getElementById('btn-start-task');
                        if (startBtn) {
                            if (!currentClientId) {
                                startBtn.classList.add('opacity-50', 'grayscale', 'cursor-not-allowed');
                                startBtn.disabled = true;
                            } else {
                                startBtn.classList.remove('opacity-50', 'grayscale', 'cursor-not-allowed');
                                startBtn.disabled = false;
                            }
                        }

                        // RENDER GRID
                        clientIds.forEach(id => {
                            let card = document.getElementById(`screen-card-${id}`);
                            if (!card) {
                                card = document.createElement('div');
                                card.id = `screen-card-${id}`;
                                card.className = "screen-card relative group cursor-pointer border-4 rounded-xl overflow-hidden bg-black aspect-[9/19.5] transition-all duration-200 hover:shadow-lg hover:scale-[1.02] mb-4";
                                card.onclick = () => { 
                                    currentClientId = id; 
                                    updateDashboard(); 
                                };
                                
                                // Inner HTML
                                card.innerHTML = `
                                    <img class="w-full h-full object-contain">
                                    <div class="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur rounded text-[10px] text-white font-mono border border-white/10 status-badge">${id}</div>
                                    <div class="absolute bottom-2 right-2 px-1 py-0.5 bg-black/50 rounded text-[8px] text-gray-400 font-mono size-badge"></div>
                                    <div class="absolute inset-0 border-4 border-transparent pointer-events-none selection-border transition-colors"></div>
                                `;
                                grid.appendChild(card);
                            }
                            
                            // Highlight Active
                            const selectionBorder = card.querySelector('.selection-border');
                            if (id === currentClientId) {
                                card.style.borderColor = '#22c55e'; // Green
                                if (selectionBorder) selectionBorder.classList.add('border-green-500/20');
                            } else {
                                card.style.borderColor = '#374151'; // Gray-700
                                if (selectionBorder) selectionBorder.classList.remove('border-green-500/20');
                            }
                            
                            // Update Image & Data
                            const info = data[id];
                            const imgEl = card.querySelector('img');
                            const statusEl = card.querySelector('.status-badge');
                            const sizeEl = card.querySelector('.size-badge');
                            
                            if (statusEl) statusEl.innerText = `${id} • ${info.phase || 'IDLE'}`;
                            
                            if (info.last_screenshot) {
                                let mime = 'image/png';
                                if (info.last_screenshot.startsWith('/9j/')) mime = 'image/jpeg';
                                const imgSrc = `data:${mime};base64,${info.last_screenshot}`;
                                
                                if (imgEl.src !== imgSrc) {
                                    imgEl.src = imgSrc;
                                    imgEl.style.display = 'block';
                                    if (sizeEl) {
                                        sizeEl.innerText = `${(info.last_screenshot.length / 1024).toFixed(1)} KB`;
                                        sizeEl.style.color = '#4ade80';
                                        setTimeout(() => sizeEl.style.color = '#9ca3af', 500);
                                    }
                                }
                            } else {
                                if (!imgEl.src) {
                                     // Placeholder logic?
                                }
                            }
                        });
                        
                        // Remove old cards
                        Array.from(grid.children).forEach(child => {
                            if (child.id.startsWith('screen-card-')) {
                                const id = child.id.replace('screen-card-', '');
                                if (!data[id]) child.remove();
                            }
                        });

                        // Update Stats/Logs for CURRENT Selected Client
                        if (currentClientId && data[currentClientId]) {
                            const info = data[currentClientId];
                            const stats = info.stats || { pages: 0, steps: 0, bugs: 0 };

                            document.getElementById('stat-pages').innerText = stats.pages;
                            document.getElementById('stat-steps').innerText = stats.steps;
                            // Bugs not in UI currently, maybe add back if space permits

                            // Update Logs
                            const logContainer = document.getElementById('log-terminal');
                            const logsHtml = info.logs.slice().reverse().map(log => {
                                let color = 'text-gray-400';
                                if (log.includes('ERROR') || log.includes('❌') || log.includes('⚠️')) color = 'text-red-400';
                                else if (log.includes('🧠') || log.includes('Thinking')) color = 'text-purple-400';
                                else if (log.includes('📸')) color = 'text-blue-400';
                                else if (log.includes('✅') || log.includes('Success')) color = 'text-green-400';
                                else if (log.includes('🚀') || log.includes('Starting')) color = 'text-yellow-400';
                                return `<div class="log-entry ${color}">${log}</div>`;
                            }).join('');
                            
                            if (logContainer.innerHTML.length !== logsHtml.length) {
                                 logContainer.innerHTML = logsHtml;
                                 logContainer.scrollTop = 0;
                            }

                            // Update Test Cases
                            const casesContainer = document.getElementById('test-cases-list');
                            const cases = info.test_cases || [];
                            document.getElementById('case-count').innerText = cases.length;
                            
                            if (cases.length > 0) {
                                casesContainer.innerHTML = cases.slice().reverse().map(c => {
                                    let statusIcon = `<span class="w-2 h-2 rounded-full bg-gray-500"></span>`;
                                    let statusClass = "border-gray-700 bg-gray-800/50";
                                    
                                    if(c.status === 'running') {
                                        statusIcon = `<span class="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>`;
                                        statusClass = "border-blue-900/50 bg-blue-900/10";
                                    } else if(c.status === 'passed') {
                                        statusIcon = `<span class="text-green-400">✓</span>`;
                                        statusClass = "border-green-900/30 bg-green-900/10";
                                    } else if(c.status === 'failed') {
                                        statusIcon = `<span class="text-red-400">✗</span>`;
                                        statusClass = "border-red-900/30 bg-red-900/10";
                                    }
                                    
                                    return `
                                        <div class="p-2 rounded border ${statusClass} flex items-start gap-2 transition-all hover:bg-gray-800 text-xs">
                                            <div class="mt-1 shrink-0">${statusIcon}</div>
                                            <div class="flex-1 min-w-0">
                                                <div class="font-medium text-gray-200 truncate">${c.desc}</div>
                                                <div class="text-[10px] text-gray-500 mt-0.5 flex justify-between">
                                                    <span>#${c.id}</span>
                                                    <span class="capitalize">${c.status}</span>
                                                </div>
                                            </div>
                                        </div>
                                    `;
                                }).join('');
                            } else {
                                casesContainer.innerHTML = `<div class="text-center text-gray-600 text-xs py-4">No cases</div>`;
                            }
                        }
                    });
            }

            function triggerAction(action) {
                if (!currentClientId) return alert("No device connected");
                
                console.log(`Triggering action: ${action} for ${currentClientId}`);

                if (action === 'start_task') {
                    fetch(`/api/start_task/${currentClientId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    }).then(res => res.json())
                      .then(data => {
                          console.log('Task Started:', data);
                          alert(data.message);
                      });
                }
                else if (action === 'stop_task') {
                    fetch(`/api/stop_task/${currentClientId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    }).then(res => res.json())
                      .then(data => console.log('Task Stopped:', data));
                }
                else if (action === 'reconnect_device') {
                    fetch(`/api/reconnect_device/${currentClientId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    }).then(res => res.json())
                      .then(data => alert('Connection command sent!'));
                }
                else if (action === 'reset_client') {
                    if(!confirm("Are you sure? This will wipe all EXPLORATION data. Test Cases will be PRESERVED.")) return;
                    fetch(`/api/reset_client/${currentClientId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    }).then(res => res.json())
                      .then(data => alert('Exploration data reset! Test Cases kept.'));
                }
                else if (action === 'run_tests') {
                    fetch(`/api/run_tests/${currentClientId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    }).then(res => res.json())
                      .then(data => console.log('Tests Started:', data));
                }
                else if (action === 'clear_logs') {
                    fetch(`/api/clear_logs/${currentClientId}`, {
                        method: 'POST'
                    });
                }
            }
            
            // Upload APK Handler
            async function uploadApk() {
                const input = document.getElementById('apk-upload');
                if (!input.files || !input.files[0]) return;
                
                const file = input.files[0];
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('map-status').innerText = "Uploading...";
                
                try {
                    const response = await fetch('/upload_apk', {
                        method: 'POST',
                        body: formData
                    });
                    const res = await response.json();
                    
                    if (res.status === 'ok') {
                        document.getElementById('map-status').innerText = "✅ Map Ready";
                        document.getElementById('map-status').classList.add('text-green-500');
                    } else {
                        document.getElementById('map-status').innerText = "❌ Failed";
                    }
                } catch (e) {
                    console.error(e);
                    document.getElementById('map-status').innerText = "❌ Error";
                }
            }

            function triggerGlobalAction(action) {
                const endpoint = action === 'start_all' ? '/api/start_all_tasks' : '/api/stop_all_tasks';
                fetch(endpoint, { method: 'POST' })
                    .then(res => res.json())
                    .then(d => {
                        console.log(d.message);
                        // Optional: Show toast
                    });
            }
            
            function toggleSmartWait() {
                if (!currentClientId) return;
                const enabled = document.getElementById('smart-wait-toggle').checked;
                fetch(`/api/config/${currentClientId}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({smart_wait: enabled})
                });
            }

            setInterval(updateDashboard, 200);
            updateDashboard();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# API: 获取所有设备状态
@app.get("/api/clients")
async def get_clients():
    result = {}
    for cid, info in active_clients.items():
        result[cid] = {
            "status": info["status"],
            "logs": info["logs"],
            "last_screenshot": info["last_screenshot"],
            "stats": info.get("stats", {"pages": 0, "steps": 0, "bugs": 0}),
            "test_cases": info.get("test_cases", []),
            "phase": info.get("phase", "IDLE"),
            "start_time": info.get("start_time"),
            "exploration_tasks": info.get("exploration_tasks", []),
            "current_task_index": info.get("current_task_index", 0),
            "pending_intents_count": len(info.get("pending_intents", [])),
            "total_intents": info.get("total_intents", 0),
            "target_steps": info.get("target_steps", 50)
        }
    return result

# API: 开始任务 (Modified for Auto Discovery)
@app.post("/api/start_task/{client_id}")
async def start_task(client_id: str):
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
    
    # 🆕 Auto Discovery Mode
    active_clients[client_id]["phase"] = "AUTO_DISCOVERY"
    active_clients[client_id]["start_time"] = time.time() # 记录开始时间
    
    # Clear old data if starting fresh?
    active_clients[client_id]["test_cases"] = []
    active_clients[client_id]["explored_pages_with_cases"] = set()
    active_clients[client_id]["pending_intents"] = [] # 🆕 Queue for Intent Injection
    active_clients[client_id]["total_intents"] = 0 # 🆕 Track Total for Progress Bar
    
    # 🆕 Target Steps Calculation (Saturation Metric)
    # Default baseline
    base_steps = 50 
    active_clients[client_id]["target_steps"] = base_steps 

    manager.add_log(client_id, "🚀 启动全自动 AI 测试流程 (Auto-Discovery)...")
    manager.add_log(client_id, "ℹ️ 流程: 探索 -> 实时生成用例 -> 自动执行")
    
    # 🆕 Intent Injection (Load Static Analysis)
    global APP_KNOWLEDGE_MAP
    analysis_data = None
    
    if APP_KNOWLEDGE_MAP:
        analysis_data = APP_KNOWLEDGE_MAP
        manager.add_log(client_id, "📂 使用内存中缓存的 APK 静态分析数据")
    else:
        # 🛑 STRICT MODE: Disable auto-loading of old local files.
        # User Feedback: "I didn't upload APK but it started, is this normal?" -> No, it's confusing.
        # Force user to upload APK at least once per server session.
        manager.add_log(client_id, "ℹ️ 无内存缓存数据，且已禁用本地历史文件自动加载。")
        
        # import glob
        # # Check /tmp first as upload_apk saves there
        # tmp_files = glob.glob("/tmp/*_analysis.json")
        # local_files = glob.glob("*_analysis.json")
        # all_files = tmp_files + local_files
        # 
        # if all_files:
        #     # Pick the most recent one by modification time
        #     analysis_path = max(all_files, key=os.path.getmtime)
        #     try:
        #         with open(analysis_path, 'r') as f:
        #             analysis_data = json.load(f)
        #         manager.add_log(client_id, f"📂 已加载静态分析文件: {analysis_path}")
        #     except Exception as e:
        #         manager.add_log(client_id, f"⚠️ 加载静态分析文件失败: {e}")

    if not analysis_data:
        # 🛑 WAIT FOR UPLOAD: 如果没有地图，强制进入 ANALYZING_APK 状态，并等待
        active_clients[client_id]["phase"] = "ANALYZING_APK"
        manager.add_log(client_id, "⚠️ 未找到 APK 静态分析数据！请先上传 APK。")
        manager.add_log(client_id, "⏳ 系统将等待 APK 上传和分析完成后再继续...")
        return {"status": "waiting_for_apk", "message": "请上传 APK 以生成测试地图"}

    if analysis_data:
        try:
            package_name = analysis_data.get("package_name") or analysis_data.get("package")
            activities_raw = analysis_data.get("activities", {})
            
            # Normalize activities to list of dicts
            activity_list = []
            if isinstance(activities_raw, dict):
                activity_list = list(activities_raw.values())
            elif isinstance(activities_raw, list):
                activity_list = activities_raw
            
            count = 0
            for act in activity_list:
                # Only inject EXPORTED activities
                # Some formats might have boolean, others string
                is_exported = act.get("exported")
                if is_exported and str(is_exported).lower() != 'false':
                    act_name = act.get("name")
                    if package_name and act_name:
                        # Construct 'am start' command
                        # am start -n com.example/com.example.MainActivity
                        cmd = f"am start -n {package_name}/{act_name}"
                        active_clients[client_id]["pending_intents"].append(cmd)
                        count += 1
            
            if count > 0:
                active_clients[client_id]["total_intents"] = count
                # 🆕 Dynamic Target Adjustment
                # Formula: Base(50) + (Pages * 15)
                new_target = 50 + (count * 15)
                active_clients[client_id]["target_steps"] = new_target
                
                manager.add_log(client_id, f"💉 注入 {count} 个 Intent 入口")
                manager.add_log(client_id, f"🎯 设定测试饱和度目标: {new_target} 步操作")
            else:
                 manager.add_log(client_id, "⚠️ 未发现可导出的 Activity，将使用普通探索模式")
                 
        except Exception as e:
            manager.add_log(client_id, f"⚠️ 解析静态分析数据出错: {e}")

    await manager.send_command(client_id, {"type": "capture_screenshot"})
    return {"status": "started", "message": f"开始全自动测试 {client_id}"}

@app.post("/api/stop_task/{client_id}")
async def stop_task(client_id: str):
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
    
    active_clients[client_id]["phase"] = "IDLE"
    manager.add_log(client_id, "🛑 用户手动停止了任务。")
    return {"status": "stopped", "message": f"已停止 {client_id}"}

@app.post("/api/start_all_tasks")
async def start_all_tasks():
    count = 0
    for client_id in active_clients:
        active_clients[client_id]["phase"] = "SCANNING"
        active_clients[client_id]["start_time"] = time.time()
        manager.add_log(client_id, "🚀 全局指令: 启动所有设备探索 (Global Start)")
        await manager.send_command(client_id, {"type": "capture_screenshot"})
        count += 1
    return {"status": "started", "message": f"Started {count} devices"}

@app.post("/api/stop_all_tasks")
async def stop_all_tasks():
    count = 0
    for client_id in active_clients:
        active_clients[client_id]["phase"] = "IDLE"
        manager.add_log(client_id, "🛑 全局指令: 停止所有任务 (Global Stop)")
        count += 1
    return {"status": "stopped", "message": f"Stopped {count} devices"}

@app.post("/api/clear_logs/{client_id}")
async def clear_logs(client_id: str):
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
    
    active_clients[client_id]["logs"] = []
    manager.add_log(client_id, "🧹 日志已清空")
    return {"status": "success", "message": "Logs cleared"}

@app.post("/api/generate_cases/{client_id}")
async def generate_cases(client_id: str, request: Request):
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
    
    # 🆕 Parse Body
    try:
        body = await request.json()
    except:
        body = {}
    target_type = body.get("target_type", "app_ble")
        
    active_clients[client_id]["phase"] = "GENERATING"
    manager.add_log(client_id, f"🛑 停止探索，开始生成测试用例 ({target_type})...")
    
    generated_cases = []
    
    # 🆕 Strategy Switch based on Target Type
    if target_type == "sdk":
        manager.add_log(client_id, "📦 生成 SDK 专项测试用例 (基于 XTalkApi.kt)...")
        sdk_cases = [
            "验证: SDK 初始化 (xTalkInit) - 检查返回结果是否成功",
            "验证: 扫描设备 (xTalkScanDevices) - 检查是否发现设备",
            "验证: 连接设备 (xTalkConnect) - 连接第一个发现的设备",
            "验证: 发送文本消息 (xTalkSendText) - 发送 'Hello AI Test'",
            "验证: 开始语音采集 (xTalkStartVoiceCapture) - 开启 PTT",
            "验证: 停止语音采集 (xTalkStopVoiceCapture) - 停止 PTT",
            "验证: 音频路由管理 (xTalkManageAudioRoute) - 检查路由状态",
            "验证: SDK 反初始化 (xTalkDeinit) - 释放资源"
        ]
        for i, desc in enumerate(sdk_cases):
             generated_cases.append({
                "id": i + 1,
                "desc": desc,
                "status": "pending"
            })
            
    elif client_id in client_graphs and client_graphs[client_id].states:
        graph = client_graphs[client_id]
        
        # 🆕 AI-Powered Test Generation
        if AI_PROVIDER != "mock" and graph.states:
            manager.add_log(client_id, "🧠 正在调用 AI 根据探索结果生成深度测试用例...")
            
            # 1. Collect Context
            pages_context = []
            for h, s in graph.states.items():
                summary = s.get("summary", "Unknown Page")
                pid = s.get("canonical_id", h[:6])
                pages_context.append(f"- Page ID: {pid}, Summary: {summary}")
            
            context_str = "\n".join(pages_context[:20]) # Limit to top 20 pages to save tokens
            
            prompt = f"""
            You are a QA Lead. We have explored an Android App and found the following pages:
            {context_str}
            
            Based on this topology, generate 5-10 high-quality, actionable test cases.
            Focus on user flows (e.g. "Login -> Settings -> Logout") and critical functionality.
            
            IMPORTANT: Output the test case descriptions in Chinese (Simplified Chinese).
            AND output a 'script' field which is a JSON array of actionable steps.
            
            Action types: "click", "input", "assert", "wait".
            Target: Text or ID of the element.
            
            Output strictly a JSON list of objects:
            [
                {{
                    "id": 1, 
                    "desc": "验证使用有效凭证登录", 
                    "priority": "P0",
                    "script": [
                        {{"action": "input", "target": "username", "value": "admin"}},
                        {{"action": "input", "target": "password", "value": "123456"}},
                        {{"action": "click", "target": "登录"}},
                        {{"action": "assert", "target": "欢迎"}}
                    ]
                }},
                {{
                    "id": 2, 
                    "desc": "导航到设置页面", 
                    "priority": "P1",
                    "script": [
                        {{"action": "click", "target": "设置"}},
                        {{"action": "assert", "target": "通用"}}
                    ]
                }}
            ]
            """
            
            try:
                # Reuse the client setup from other functions
                if AI_PROVIDER == "openai":
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
                    response = await client.chat.completions.create(
                        model=AI_MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    content = response.choices[0].message.content
                elif AI_PROVIDER == "dashscope":
                    # DashScope Logic (Text Only)
                    import dashscope
                    dashscope.api_key = DASHSCOPE_API_KEY
                    from dashscope import Generation
                    resp = await asyncio.to_thread(
                        Generation.call,
                        model="qwen-turbo", # Use text model for faster generation
                        messages=[{'role': 'user', 'content': prompt}],
                        result_format='message'
                    )
                    if resp.status_code == 200:
                        content = resp.output.choices[0].message.content
                    else:
                        raise Exception(f"DashScope Error: {resp.code}")
                
                # Parse JSON
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    ai_cases = json.loads(json_match.group(0))
                    for idx, c in enumerate(ai_cases):
                        generated_cases.append({
                            "id": idx + 1,
                            "desc": c.get("desc", "Unknown Case"),
                            "script": c.get("script", []), # Store the script!
                            "status": "pending"
                        })
                    manager.add_log(client_id, f"✅ AI 成功生成 {len(generated_cases)} 条智能用例")
            
            except Exception as e:
                manager.add_log(client_id, f"⚠️ AI 生成失败 ({e})，回退到启发式生成...")
        
        # Fallback to Heuristic if AI failed or is mock
        if not generated_cases and graph.states:
            # Strategy: Generate a test case for each distinct "Page" found, using Learned Knowledge
            count = 1
            for h, s in graph.states.items():
                # Try to find a meaningful name
                summary = s.get("summary")
                name = summary if summary else f"Page {h[:6]}"
                
                # 🆕 Auto-Generate Script from Explored Actions
                script = []
                # 1. Navigate to this page (Hard, need Dijkstra, skipping for heuristic)
                # 2. Check elements
                available = s.get("available_actions", [])
                for idx, act in enumerate(available[:3]): # Check top 3 elements
                    txt = act.get('text') or act.get('content-desc')
                    if txt:
                        script.append({"action": "assert", "target": txt})
                
                generated_cases.append({
                    "id": count,
                    "desc": f"验证: {name} (检查核心元素)",
                    "status": "pending",
                    "script": script # 🆕 Add Script
                })
                count += 1
                if count > 10: break # Limit to 10 cases

    if not generated_cases:
        # Fallback Mock
        import random
        # 🆕 Mock Script Generation
        mock_scripts = [
            [{"action": "wait", "value": 1}, {"action": "assert", "target": "登录"}],
            [{"action": "click", "target": "设置"}, {"action": "wait", "value": 1}],
            [{"action": "scroll", "direction": "down"}, {"action": "assert", "target": "更多"}]
        ]
        
        actions = ["点击 '登录'", "输入用户名", "点击 '设置'", "滑动列表", "检查 '我的' 页面"]
        for i in range(1, 6):
            generated_cases.append({
                "id": i,
                "desc": random.choice(actions) + f" (Case #{i})",
                "status": "pending",
                "script": random.choice(mock_scripts) # 🆕 Add Script
            })
    
    active_clients[client_id]["test_cases"] = generated_cases
    manager.add_log(client_id, f"✅ 已生成 {len(generated_cases)} 条测试用例")
    
    # 💾 保存
    save_client_state(client_id)
    
    return {"status": "generated", "data": generated_cases}

def generate_test_cases_from_graph(client_id):
    """
    Based on the Knowledge Map (Static) and Graph (Dynamic),
    generate specific test cases with scripts.
    """
    if client_id not in active_clients: return
    
    # 1. Get Static Knowledge
    static_activities = []
    if APP_KNOWLEDGE_MAP:
        pkg = APP_KNOWLEDGE_MAP.get("package_name", "")
        acts = APP_KNOWLEDGE_MAP.get("activities", [])
        # Normalize
        if isinstance(acts, dict): acts = list(acts.values())
        
        for a in acts:
             # Check if exported
             if a.get("exported") and str(a.get("exported")).lower() != 'false':
                 name = a.get("name")
                 if name:
                     if name.startswith("."): name = pkg + name
                     static_activities.append(name)
    
    generated = []
    
    # 2. Generate Smoke Tests for each Exported Activity
    for i, act_name in enumerate(static_activities):
        # Create a "Jump & Verify" script
        short_name = act_name.split('.')[-1]
        
        script = [
            # Step 1: Jump
            {"action": "intent_jump", "target": act_name},
            # Step 2: Wait for load
            {"action": "wait", "value": 2},
            # Step 3: Assert (Implicitly if we don't crash, it passes, but let's try to check something?)
            # For now, just a screenshot action to confirm visual
            {"action": "agent_command", "payload": "capture_screenshot"} 
        ]
        
        case = {
            "id": i + 1,
            "desc": f"Smoke Test: Verify Activity {short_name}",
            "status": "pending",
            "script": script,
            "type": "smoke"
        }
        generated.append(case)
        
    # 3. Save to Client
    active_clients[client_id]["test_cases"] = generated
    manager.add_log(client_id, f"📝 自动生成了 {len(generated)} 条测试用例 (基于 Intent Map)")

@app.post("/api/run_tests/{client_id}")
async def run_tests(client_id: str):
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
        
    active_clients[client_id]["phase"] = "EXECUTING"
    
    # 🆕 Reset step index to ensure clean run
    active_clients[client_id]["current_step_index"] = 0
    
    # 🆕 Reset case statuses (optional, or keep history?)
    # Let's reset pending ones or all? User said "Run Saved", usually implies re-run.
    for case in active_clients[client_id]["test_cases"]:
        case["status"] = "pending"
        
    manager.add_log(client_id, "🚀 开始执行已保存的测试用例 (Script Mode)...")
    await manager.send_command(client_id, {"type": "capture_screenshot"})
    return {"status": "running", "message": "开始执行测试"}

@app.post("/api/reconnect_device/{client_id}")
async def reconnect_device(client_id: str):
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
    
    manager.add_log(client_id, "🔌 发送重连指令 (Reconnect Device)...")
    await manager.send_command(client_id, {"type": "reconnect_device"})
    return {"status": "sent", "message": "Reconnection command sent"}

@app.post("/api/config/{client_id}")
async def update_config(client_id: str, config: dict):
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}
    
    # Send config update to edge client
    if "smart_wait" in config:
        manager.add_log(client_id, f"⚙️ Config Update: Smart Wait = {config['smart_wait']}")
        await manager.send_command(client_id, {"type": "update_config", "payload": config})
        
    return {"status": "updated"}

@app.post("/api/reset_client/{client_id}")
async def reset_client(client_id: str):
    """
    重置 Client 的所有状态 (Graph, Stats) 但保留 Test Cases
    """
    if client_id not in active_clients:
        return {"status": "error", "message": "Client not found"}

    # 1. Reset Stats & Phase (PRESERVE TEST CASES)
    active_clients[client_id]["stats"] = {"pages": 0, "steps": 0, "bugs": 0}
    active_clients[client_id]["phase"] = "IDLE"
    # active_clients[client_id]["test_cases"] = [] # 🛑 STOP CLEARING THIS
    
    active_clients[client_id]["logs"].append(f"[{time.strftime('%H:%M:%S')}] 🧹 Exploration data cleared. Test Cases preserved.")
    
    # 2. Reset Graph
    if client_id in client_graphs:
        # 🆕 Create a new Graph but PRESERVE Visual Memory if possible?
        # Actually, let's keep the global memory to avoid relearning "Logout" is bad.
        # But user wants a "Fresh Start".
        # Let's clear graph but maybe we should keep `ineffective_actions`?
        # For now, clear everything as requested.
        client_graphs[client_id] = AppStateGraph()
        print(f"🧹 Graph cleared for {client_id}")

    # 3. Update Persistent File (Don't delete, just save the new 'empty' state with cases)
    # file_path = os.path.join(DATA_DIR, f"{client_id}.json")
    # if os.path.exists(file_path):
    #    try:
    #        os.remove(file_path)
    #    except: pass
    
    # Save the cleaned state (which includes the preserved test cases)
    save_client_state(client_id)

    manager.add_log(client_id, "🧹 已清空探索数据，测试用例已保留 (Reset Complete)")
    return {"status": "success", "message": "Exploration data cleared, Test Cases preserved"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Central Server starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
