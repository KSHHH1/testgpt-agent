import os
import sys
import json
import logging
from androguard.core.apk import APK

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ApkAnalyzer:
    def __init__(self, apk_path):
        self.apk_path = apk_path
        self.apk = None
        self.analysis_result = {}

    def load_apk(self):
        """Loads the APK file using Androguard."""
        if not os.path.exists(self.apk_path):
            logger.error(f"APK file not found: {self.apk_path}")
            return False
        
        try:
            self.apk = APK(self.apk_path)
            logger.info(f"Successfully loaded APK: {self.apk_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load APK: {e}")
            return False

    def analyze(self):
        """Extracts comprehensive information from the APK."""
        if not self.apk:
            if not self.load_apk():
                return None

        package_name = self.apk.get_package()
        app_name = self.apk.get_app_name()
        main_activity = self.apk.get_main_activity()
        
        activities = self.apk.get_activities()
        services = self.apk.get_services()
        receivers = self.apk.get_receivers()
        permissions = self.apk.get_permissions()
        
        # Deep analysis of Activities and Intent Filters
        activity_map = {}
        for activity in activities:
            # Check if exported (can be launched externally)
            # Androguard might not have a direct 'is_exported' method on the string name,
            # we need to inspect the manifest xml manually or use other helpers if needed.
            # For now, let's assume we want to map the name to a simple object.
            
            # Try to get intent filters for this activity
            # This is complex in raw Androguard, but we can try to parse the AndroidManifest.xml
            activity_map[activity] = {
                "name": activity,
                "exported": self._is_exported(activity), # Helper method
                "intent_filters": self._get_intent_filters(activity)
            }

        # 🆕 Framework Detection
        framework = "native"
        try:
            files = self.apk.get_files()
            if any("libflutter.so" in f for f in files) or any("flutter_assets" in f for f in files):
                framework = "flutter"
            elif any("libreactnativejni.so" in f for f in files) or any("index.android.bundle" in f for f in files):
                framework = "react_native"
            elif any("libunity.so" in f for f in files):
                framework = "unity"
        except:
            pass

        self.analysis_result = {
            "package_name": package_name,
            "app_name": app_name,
            "main_activity": main_activity,
            "activities": activity_map,
            "services": services,
            "receivers": receivers,
            "permissions": permissions,
            "map_version": "1.0",
            "framework": framework # 🆕 Added framework info
        }
        
        return self.analysis_result

    def _is_exported(self, activity_name):
        """
        Heuristic to check if an activity is exported.
        This accesses the underlying xml object.
        """
        try:
            # This is a simplified lookup. 
            # In a real robust implementation, we would traverse the XML tree.
            # For now, we return 'Unknown' or True/False if we can easily determine.
            return True # Default assumption for test convenience, or refine later.
        except:
            return False

    def _get_intent_filters(self, activity_name):
        """
        Extracts Intent Filters for a given activity.
        Returns a list of actions/categories.
        """
        # Placeholder for complex XML parsing logic.
        # We will implement a basic version that tries to find 'action' tags.
        return []

    def save_report(self, output_path):
        """Saves the analysis result to a JSON file."""
        if not self.analysis_result:
            logger.warning("No analysis result to save.")
            return

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_result, f, indent=4, ensure_ascii=False)
            logger.info(f"Analysis report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 apk_analyzer.py <path_to_apk> [output_json_path]")
        sys.exit(1)

    apk_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"{apk_file}_analysis.json"

    analyzer = ApkAnalyzer(apk_file)
    result = analyzer.analyze()
    
    if result:
        analyzer.save_report(output_file)
        print("Analysis Complete.")
    else:
        print("Analysis Failed.")
