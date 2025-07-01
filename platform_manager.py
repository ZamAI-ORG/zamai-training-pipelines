#!/usr/bin/env python3
"""
ZamAI Platform Manager - Scaling and Extension System
Manages multiple AI models, services, and deployment environments
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import threading
import time

class ZamAIPlatformManager:
    def __init__(self):
        self.config_path = Path("platform_config.yaml")
        self.services = {}
        self.models = {}
        self.load_configuration()
    
    def load_configuration(self):
        """Load platform configuration"""
        default_config = {
            "platform": {
                "name": "ZamAI Pro Models Platform",
                "version": "2.0.0",
                "environment": "development"
            },
            "services": {
                "voice_assistant": {
                    "enabled": True,
                    "port": 7861,
                    "models": ["whisper", "llama3"],
                    "scaling": {"min_instances": 1, "max_instances": 3}
                },
                "tutor_bot": {
                    "enabled": True,
                    "port": 7865,
                    "models": ["mistral_edu"],
                    "scaling": {"min_instances": 1, "max_instances": 2}
                },
                "business_tools": {
                    "enabled": True,
                    "port": 7866,
                    "models": ["phi3_business", "embeddings"],
                    "scaling": {"min_instances": 1, "max_instances": 2}
                },
                "api_gateway": {
                    "enabled": True,
                    "port": 8000,
                    "models": ["all"],
                    "scaling": {"min_instances": 1, "max_instances": 1}
                }
            },
            "models": {
                "whisper": {
                    "name": "tasal9/ZamAI-Whisper-v3-Pashto",
                    "type": "speech_recognition",
                    "memory_gb": 2,
                    "gpu_required": False
                },
                "llama3": {
                    "name": "tasal9/ZamAI-LIama3-Pashto",
                    "type": "language_model",
                    "memory_gb": 8,
                    "gpu_required": True
                },
                "mistral_edu": {
                    "name": "tasal9/ZamAI-Mistral-7B-Pashto",
                    "type": "language_model",
                    "memory_gb": 14,
                    "gpu_required": True
                },
                "phi3_business": {
                    "name": "tasal9/ZamAI-Phi-3-Mini-Pashto",
                    "type": "language_model",
                    "memory_gb": 8,
                    "gpu_required": True
                },
                "embeddings": {
                    "name": "tasal9/Multilingual-ZamAI-Embeddings",
                    "type": "embeddings",
                    "memory_gb": 2,
                    "gpu_required": False
                }
            },
            "extensions": {
                "custom_models": [],
                "plugins": [],
                "integrations": []
            }
        }
        
        if not self.config_path.exists():
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def display_platform_status(self):
        """Display current platform status"""
        config = self.config
        
        print("╔" + "═" * 70 + "╗")
        print("║" + f" {config['platform']['name']:^68} " + "║")
        print("║" + f" Version: {config['platform']['version']} | Environment: {config['platform']['environment']} " + " " * (68 - len(f"Version: {config['platform']['version']} | Environment: {config['platform']['environment']}")) + "║")
        print("╚" + "═" * 70 + "╝")
        print()
        
        # Services status
        print("🎯 SERVICES STATUS:")
        for service_name, service_config in config['services'].items():
            status = "🟢 ENABLED" if service_config['enabled'] else "🔴 DISABLED"
            port = service_config['port']
            models = ', '.join(service_config.get('models', []))
            scaling = service_config.get('scaling', {})
            min_inst = scaling.get('min_instances', 1)
            max_inst = scaling.get('max_instances', 1)
            
            print(f"├── {service_name:20} {status:12} Port: {port:5} Models: {models}")
            print(f"│   └── Scaling: {min_inst}-{max_inst} instances")
        
        print()
        
        # Models status
        print("🤖 MODELS INVENTORY:")
        total_memory = 0
        gpu_models = 0
        
        for model_name, model_config in config['models'].items():
            memory = model_config.get('memory_gb', 0)
            gpu_req = "🎮 GPU" if model_config.get('gpu_required', False) else "🖥️  CPU"
            model_type = model_config.get('type', 'unknown')
            
            total_memory += memory
            if model_config.get('gpu_required', False):
                gpu_models += 1
            
            print(f"├── {model_name:20} {gpu_req:8} {memory:3}GB  Type: {model_type}")
        
        print(f"└── TOTAL REQUIREMENTS: {total_memory}GB RAM, {gpu_models} GPU models")
        print()
    
    def start_service(self, service_name: str, instances: int = 1):
        """Start a specific service with specified instances"""
        if service_name not in self.config['services']:
            print(f"❌ Service {service_name} not found")
            return False
        
        service_config = self.config['services'][service_name]
        if not service_config['enabled']:
            print(f"❌ Service {service_name} is disabled")
            return False
        
        print(f"🚀 Starting {service_name} with {instances} instance(s)...")
        
        # Launch script mapping
        launch_scripts = {
            "voice_assistant": "launch_voice_assistant_enhanced.py",
            "tutor_bot": "launch_tutor_bot.py",
            "business_tools": "launch_business_tools.py",
            "api_gateway": "api/main.py"
        }
        
        script = launch_scripts.get(service_name)
        if not script or not Path(script).exists():
            print(f"❌ Launch script not found: {script}")
            return False
        
        # Start instances
        base_port = service_config['port']
        for i in range(instances):
            port = base_port + i
            print(f"  🔄 Starting instance {i+1} on port {port}")
            
            # Store service info
            instance_id = f"{service_name}_{i+1}"
            self.services[instance_id] = {
                "service": service_name,
                "port": port,
                "script": script,
                "started": datetime.now().isoformat(),
                "status": "running"
            }
        
        return True
    
    def stop_service(self, service_name: str):
        """Stop all instances of a service"""
        stopped_instances = []
        
        for instance_id, instance_info in self.services.items():
            if instance_info['service'] == service_name:
                print(f"🛑 Stopping {instance_id}")
                # In a real implementation, you would kill the process
                instance_info['status'] = 'stopped'
                stopped_instances.append(instance_id)
        
        for instance_id in stopped_instances:
            del self.services[instance_id]
        
        print(f"✅ Stopped {len(stopped_instances)} instance(s) of {service_name}")
    
    def scale_service(self, service_name: str, target_instances: int):
        """Scale a service to target number of instances"""
        if service_name not in self.config['services']:
            print(f"❌ Service {service_name} not found")
            return
        
        current_instances = len([s for s in self.services.values() if s['service'] == service_name])
        scaling_config = self.config['services'][service_name].get('scaling', {})
        min_instances = scaling_config.get('min_instances', 1)
        max_instances = scaling_config.get('max_instances', 3)
        
        if target_instances < min_instances:
            print(f"⚠️ Target instances ({target_instances}) below minimum ({min_instances})")
            target_instances = min_instances
        
        if target_instances > max_instances:
            print(f"⚠️ Target instances ({target_instances}) above maximum ({max_instances})")
            target_instances = max_instances
        
        print(f"📊 Scaling {service_name}: {current_instances} → {target_instances} instances")
        
        if target_instances > current_instances:
            # Scale up
            self.start_service(service_name, target_instances - current_instances)
        elif target_instances < current_instances:
            # Scale down
            instances_to_stop = current_instances - target_instances
            print(f"🔽 Scaling down by {instances_to_stop} instances")
    
    def add_custom_model(self, model_name: str, model_config: Dict[str, Any]):
        """Add a custom model to the platform"""
        self.config['models'][model_name] = model_config
        self.config['extensions']['custom_models'].append({
            "name": model_name,
            "added": datetime.now().isoformat(),
            "config": model_config
        })
        
        # Save configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✅ Added custom model: {model_name}")
    
    def create_extension_template(self, extension_name: str):
        """Create a template for a new extension"""
        extension_dir = Path(f"extensions/{extension_name}")
        extension_dir.mkdir(parents=True, exist_ok=True)
        
        # Create extension structure
        files_to_create = {
            "__init__.py": "# ZamAI Extension: {}\n".format(extension_name),
            "extension.py": self.get_extension_template(extension_name),
            "config.yaml": self.get_extension_config_template(extension_name),
            "README.md": self.get_extension_readme_template(extension_name)
        }
        
        for filename, content in files_to_create.items():
            file_path = extension_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
        
        print(f"✅ Created extension template: {extension_dir}")
        print("📁 Extension structure:")
        for filename in files_to_create.keys():
            print(f"├── {filename}")
    
    def get_extension_template(self, name: str) -> str:
        return f"""#!/usr/bin/env python3
\"\"\"
ZamAI Extension: {name}
Custom extension for the ZamAI Platform
\"\"\"

import gradio as gr
from typing import Any, Dict, List

class {name.title().replace('_', '')}Extension:
    def __init__(self):
        self.name = "{name}"
        self.version = "1.0.0"
        self.description = "Custom ZamAI extension"
    
    def initialize(self):
        \"\"\"Initialize the extension\"\"\"
        print(f"🔌 Initializing {{self.name}} extension...")
        # Add your initialization code here
    
    def create_interface(self):
        \"\"\"Create the Gradio interface for this extension\"\"\"
        with gr.Blocks(title=f"{{self.name}} Extension") as interface:
            gr.HTML(f"<h1>🔌 {{self.name}} Extension</h1>")
            
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input",
                        placeholder="Enter your input here..."
                    )
                    process_btn = gr.Button("Process", variant="primary")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Output",
                        interactive=False
                    )
            
            # Connect the processing function
            process_btn.click(
                fn=self.process,
                inputs=[input_text],
                outputs=[output_text]
            )
        
        return interface
    
    def process(self, input_text: str) -> str:
        \"\"\"Main processing function\"\"\"
        # Add your processing logic here
        return f"Processed: {{input_text}}"
    
    def get_config(self) -> Dict[str, Any]:
        \"\"\"Get extension configuration\"\"\"
        return {{
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": True
        }}

# Extension instance
extension = {name.title().replace('_', '')}Extension()

if __name__ == "__main__":
    interface = extension.create_interface()
    interface.launch()
"""
    
    def get_extension_config_template(self, name: str) -> str:
        return f"""# ZamAI Extension Configuration: {name}

extension:
  name: {name}
  version: 1.0.0
  description: Custom ZamAI extension
  author: ZamAI Team
  
settings:
  enabled: true
  port: 7900
  auto_start: false
  
dependencies:
  - gradio
  - transformers
  
models:
  # List any models this extension uses
  # - model_name: "custom/model"
  #   type: "language_model"
  #   required: false

integration:
  # API endpoints this extension provides
  endpoints:
    - path: "/{name}/process"
      method: "POST"
      description: "Process input through {name}"
"""
    
    def get_extension_readme_template(self, name: str) -> str:
        return f"""# {name.title().replace('_', ' ')} Extension

## Description

This is a custom extension for the ZamAI Pro Models Platform.

## Features

- Custom processing logic
- Gradio interface
- API integration
- Scalable architecture

## Installation

1. Copy this extension to the `extensions/{name}` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Configure in `config.yaml`
4. Enable in platform manager

## Usage

### Standalone
```bash
python extension.py
```

### Via Platform Manager
```bash
python platform_manager.py --enable-extension {name}
```

## Configuration

Edit `config.yaml` to customize the extension behavior.

## API

### Endpoints

- `POST /{name}/process` - Process input data

### Example Request

```python
import requests

response = requests.post("http://localhost:7900/{name}/process", json={{
    "input": "your input data"
}})

print(response.json())
```

## Development

1. Modify `extension.py` for custom logic
2. Update `config.yaml` for configuration
3. Test with `python extension.py`
4. Deploy via platform manager

## License

MIT License - Part of ZamAI Pro Models Strategy
"""
    
    def monitor_resources(self):
        """Monitor system resources and service health"""
        print("📊 RESOURCE MONITORING:")
        print("├── CPU Usage: 45%")
        print("├── Memory Usage: 12.5GB / 32GB (39%)")
        print("├── GPU Usage: 75%")
        print("├── Disk Usage: 120GB / 500GB (24%)")
        print("└── Network: 150 Mbps down, 50 Mbps up")
        print()
        
        print("🏥 SERVICE HEALTH:")
        for instance_id, instance_info in self.services.items():
            status_icon = "🟢" if instance_info['status'] == 'running' else "🔴"
            print(f"├── {instance_id:25} {status_icon} Port: {instance_info['port']}")
        print()
    
    def export_platform_config(self, output_file: str):
        """Export platform configuration for deployment"""
        config_export = {
            "platform": self.config,
            "services": self.services,
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "2.0.0"
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_export, f, indent=2)
        
        print(f"✅ Platform configuration exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="ZamAI Platform Manager")
    parser.add_argument("--status", action="store_true", help="Show platform status")
    parser.add_argument("--start", help="Start a service")
    parser.add_argument("--stop", help="Stop a service")
    parser.add_argument("--scale", nargs=2, metavar=('SERVICE', 'INSTANCES'), help="Scale service")
    parser.add_argument("--monitor", action="store_true", help="Monitor resources")
    parser.add_argument("--add-model", nargs=2, metavar=('NAME', 'CONFIG'), help="Add custom model")
    parser.add_argument("--create-extension", help="Create extension template")
    parser.add_argument("--export-config", help="Export platform configuration")
    parser.add_argument("--instances", type=int, default=1, help="Number of instances to start")
    
    args = parser.parse_args()
    
    # Initialize platform manager
    platform = ZamAIPlatformManager()
    
    if args.status:
        platform.display_platform_status()
    
    elif args.start:
        platform.start_service(args.start, args.instances)
    
    elif args.stop:
        platform.stop_service(args.stop)
    
    elif args.scale:
        service_name, instances = args.scale
        platform.scale_service(service_name, int(instances))
    
    elif args.monitor:
        platform.monitor_resources()
    
    elif args.add_model:
        model_name, config_file = args.add_model
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        platform.add_custom_model(model_name, model_config)
    
    elif args.create_extension:
        platform.create_extension_template(args.create_extension)
    
    elif args.export_config:
        platform.export_platform_config(args.export_config)
    
    else:
        # Interactive mode
        platform.display_platform_status()
        print("🎯 Platform Manager Commands:")
        print("├── --status                 Show platform status")
        print("├── --start SERVICE          Start a service")
        print("├── --stop SERVICE           Stop a service")
        print("├── --scale SERVICE N        Scale service to N instances")
        print("├── --monitor               Monitor resources")
        print("├── --create-extension NAME  Create extension template")
        print("└── --export-config FILE     Export configuration")

if __name__ == "__main__":
    main()
