#!/usr/bin/env python3

import gradio as gr
import subprocess
import json
import os
import time
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd

class ZamAIDeploymentDashboard:
    def __init__(self):
        self.models = {
            "ZamAI-Whisper-v3-Pashto": "tasal9/ZamAI-Whisper-v3-Pashto",
            "ZamAI-Mistral-7B-Pashto": "tasal9/ZamAI-Mistral-7B-Pashto", 
            "ZamAI-Phi-3-Mini-Pashto": "tasal9/ZamAI-Phi-3-Mini-Pashto",
            "ZamAI-LIama3-Pashto": "tasal9/ZamAI-LIama3-Pashto",
            "Multilingual-ZamAI-Embeddings": "tasal9/Multilingual-ZamAI-Embeddings",
            "pashto-base-bloom": "tasal9/pashto-base-bloom"
        }
        
        self.services = {
            "Voice Assistant": {"port": 7861, "script": "launch_voice_assistant.py"},
            "Tutor Bot": {"port": 7865, "script": "launch_tutor_bot.py"},
            "Business Tools": {"port": 7866, "script": "launch_business_tools.py"},
            "API Gateway": {"port": 8000, "script": "api/main.py"}
        }
        
        self.deployment_log = []
    
    def check_model_status(self, model_repo):
        """Check HuggingFace model status"""
        try:
            url = f"https://huggingface.co/api/models/{model_repo}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "✅ Available",
                    "downloads": data.get("downloads", 0),
                    "last_modified": data.get("lastModified", "Unknown")[:10],
                    "private": "🔒 Private" if data.get("private", False) else "🌐 Public"
                }
            elif response.status_code == 403:
                return {"status": "🔒 Private/Auth Required", "downloads": "N/A", "last_modified": "N/A", "private": "🔒 Private"}
            else:
                return {"status": "❌ Unavailable", "downloads": "N/A", "last_modified": "N/A", "private": "❌ Error"}
        except:
            return {"status": "🔍 Checking...", "downloads": "N/A", "last_modified": "N/A", "private": "N/A"}
    
    def get_models_status(self):
        """Get status of all ZamAI models"""
        results = []
        for model_name, model_repo in self.models.items():
            status = self.check_model_status(model_repo)
            results.append([
                model_name,
                model_repo,
                status["status"],
                status["downloads"],
                status["last_modified"],
                status["private"]
            ])
        
        df = pd.DataFrame(results, columns=[
            "Model Name", "Repository", "Status", "Downloads", "Last Modified", "Visibility"
        ])
        return df
    
    def check_service_health(self, port):
        """Check if service is running on port"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return "🟢 Running" if response.status_code == 200 else "🟡 Unhealthy"
        except:
            return "🔴 Stopped"
    
    def get_services_status(self):
        """Get status of all services"""
        results = []
        for service_name, config in self.services.items():
            port = config["port"]
            status = self.check_service_health(port)
            url = f"http://localhost:{port}"
            
            results.append([
                service_name,
                port,
                status,
                url,
                config["script"]
            ])
        
        df = pd.DataFrame(results, columns=[
            "Service", "Port", "Status", "URL", "Script"
        ])
        return df
    
    def start_service(self, service_name):
        """Start a service"""
        if service_name not in self.services:
            return f"❌ Unknown service: {service_name}"
        
        config = self.services[service_name]
        script = config["script"]
        port = config["port"]
        
        try:
            # Check if already running
            if self.check_service_health(port) == "🟢 Running":
                return f"ℹ️ {service_name} is already running on port {port}"
            
            # Start the service
            cmd = [
                "python", script,
                "--host", "0.0.0.0", 
                "--port", str(port),
                "--background"
            ]
            
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait and check
            time.sleep(3)
            status = self.check_service_health(port)
            
            log_entry = f"{datetime.now().strftime('%H:%M:%S')} - Started {service_name} on port {port}"
            self.deployment_log.append(log_entry)
            
            if status == "🟢 Running":
                return f"✅ {service_name} started successfully on port {port}"
            else:
                return f"⚠️ {service_name} may have issues starting"
                
        except Exception as e:
            error_msg = f"❌ Error starting {service_name}: {str(e)}"
            self.deployment_log.append(error_msg)
            return error_msg
    
    def stop_service(self, service_name):
        """Stop a service"""
        if service_name not in self.services:
            return f"❌ Unknown service: {service_name}"
        
        port = self.services[service_name]["port"]
        
        try:
            # Kill process on port (Unix-like systems)
            subprocess.run(["pkill", "-f", f"port {port}"], capture_output=True)
            
            log_entry = f"{datetime.now().strftime('%H:%M:%S')} - Stopped {service_name}"
            self.deployment_log.append(log_entry)
            
            return f"🛑 {service_name} stopped"
            
        except Exception as e:
            error_msg = f"❌ Error stopping {service_name}: {str(e)}"
            self.deployment_log.append(error_msg)
            return error_msg
    
    def deploy_docker_stack(self):
        """Deploy the full Docker stack"""
        try:
            log_entry = f"{datetime.now().strftime('%H:%M:%S')} - Starting Docker deployment"
            self.deployment_log.append(log_entry)
            
            # Check if Docker Compose file exists
            if not Path("docker-compose.production.yml").exists():
                return "❌ docker-compose.production.yml not found"
            
            # Start Docker stack
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml", 
                "up", "-d", "--build"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                log_entry = f"{datetime.now().strftime('%H:%M:%S')} - Docker stack deployed successfully"
                self.deployment_log.append(log_entry)
                return "✅ Docker stack deployed successfully!\n\nServices available at:\n- Main Dashboard: http://localhost\n- API Gateway: http://localhost:8000\n- Voice Assistant: http://localhost:8001\n- Tutor Bot: http://localhost:8002\n- Business Tools: http://localhost:8003"
            else:
                error_msg = f"❌ Docker deployment failed: {result.stderr}"
                self.deployment_log.append(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"❌ Docker deployment error: {str(e)}"
            self.deployment_log.append(error_msg)
            return error_msg
    
    def get_deployment_log(self):
        """Get recent deployment log entries"""
        if not self.deployment_log:
            return "No deployment activities yet."
        
        # Return last 20 entries
        recent_logs = self.deployment_log[-20:]
        return "\n".join(recent_logs)
    
    def create_hf_space(self, space_name, space_type):
        """Create HuggingFace Space"""
        try:
            spaces = {
                "Voice Assistant": "hf_spaces/voice-assistant",
                "Business Tools": "hf_spaces/business-tools", 
                "Tutor Bot": "hf_space"
            }
            
            if space_type not in spaces:
                return f"❌ Unknown space type: {space_type}"
            
            space_dir = Path(spaces[space_type])
            if not space_dir.exists():
                return f"❌ Space directory not found: {space_dir}"
            
            log_entry = f"{datetime.now().strftime('%H:%M:%S')} - Creating HuggingFace Space: {space_name}"
            self.deployment_log.append(log_entry)
            
            return f"✅ HuggingFace Space configuration ready!\n\nTo deploy:\n1. Create space on HuggingFace Hub: https://huggingface.co/new-space\n2. Clone the space repository\n3. Copy files from {space_dir}/\n4. Push to the space repository\n\nSpace will be available at: https://huggingface.co/spaces/tasal9/{space_name}"
            
        except Exception as e:
            error_msg = f"❌ Error creating HF Space: {str(e)}"
            self.deployment_log.append(error_msg)
            return error_msg

# Initialize dashboard
dashboard = ZamAIDeploymentDashboard()

# Create Gradio interface
def create_dashboard():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="ZamAI Deployment Dashboard",
        css="""
        .gradio-container { max-width: 1400px !important; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-card { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .warning { color: #ffc107; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🚀 ZamAI Deployment Dashboard</h1>
            <p>Comprehensive platform management for production deployment</p>
        </div>
        """)
        
        with gr.Tabs():
            # Models Status Tab
            with gr.TabItem("🤖 Models Status"):
                gr.HTML("<h3>📊 HuggingFace Models Status</h3>")
                
                models_refresh_btn = gr.Button("🔄 Refresh Models Status", variant="secondary")
                models_status_df = gr.Dataframe(
                    headers=["Model Name", "Repository", "Status", "Downloads", "Last Modified", "Visibility"],
                    value=dashboard.get_models_status(),
                    interactive=False
                )
                
                models_refresh_btn.click(
                    fn=dashboard.get_models_status,
                    outputs=[models_status_df]
                )
            
            # Services Management Tab
            with gr.TabItem("🔧 Services Management"):
                gr.HTML("<h3>🎯 Local Services Control</h3>")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        service_dropdown = gr.Dropdown(
                            choices=list(dashboard.services.keys()),
                            label="Select Service",
                            value="Voice Assistant"
                        )
                        
                        with gr.Row():
                            start_btn = gr.Button("▶️ Start Service", variant="primary")
                            stop_btn = gr.Button("⏹️ Stop Service", variant="secondary")
                        
                        service_result = gr.Textbox(
                            label="Service Control Result",
                            lines=3,
                            placeholder="Service control results will appear here..."
                        )
                    
                    with gr.Column(scale=1):
                        services_refresh_btn = gr.Button("🔄 Refresh Services Status", variant="secondary")
                        services_status_df = gr.Dataframe(
                            headers=["Service", "Port", "Status", "URL", "Script"],
                            value=dashboard.get_services_status(),
                            interactive=False
                        )
                
                start_btn.click(
                    fn=dashboard.start_service,
                    inputs=[service_dropdown],
                    outputs=[service_result]
                )
                
                stop_btn.click(
                    fn=dashboard.stop_service,
                    inputs=[service_dropdown],
                    outputs=[service_result]
                )
                
                services_refresh_btn.click(
                    fn=dashboard.get_services_status,
                    outputs=[services_status_df]
                )
            
            # Production Deployment Tab
            with gr.TabItem("🐳 Production Deployment"):
                gr.HTML("<h3>🚀 Docker Production Deployment</h3>")
                
                deploy_docker_btn = gr.Button("🐳 Deploy Docker Stack", variant="primary", size="lg")
                docker_result = gr.Textbox(
                    label="Deployment Result",
                    lines=10,
                    placeholder="Docker deployment results will appear here..."
                )
                
                gr.HTML("""
                <div class="status-card">
                    <h4>📋 Production Stack Includes:</h4>
                    <ul>
                        <li>🎤 Voice Assistant Service (Port 8001)</li>
                        <li>🎓 Tutor Bot Service (Port 8002)</li>
                        <li>📄 Business Tools Service (Port 8003)</li>
                        <li>🔌 API Gateway (Port 8000)</li>
                        <li>🗄️ Redis Cache</li>
                        <li>⚖️ Nginx Load Balancer (Port 80/443)</li>
                    </ul>
                </div>
                """)
                
                deploy_docker_btn.click(
                    fn=dashboard.deploy_docker_stack,
                    outputs=[docker_result]
                )
            
            # HuggingFace Spaces Tab
            with gr.TabItem("🤗 HuggingFace Spaces"):
                gr.HTML("<h3>🌐 HuggingFace Spaces Deployment</h3>")
                
                with gr.Row():
                    space_name = gr.Textbox(
                        label="Space Name",
                        placeholder="zamai-voice-assistant",
                        value="zamai-voice-assistant"
                    )
                    
                    space_type = gr.Dropdown(
                        choices=["Voice Assistant", "Business Tools", "Tutor Bot"],
                        label="Space Type",
                        value="Voice Assistant"
                    )
                
                create_space_btn = gr.Button("🤗 Prepare HF Space", variant="primary")
                space_result = gr.Textbox(
                    label="Space Creation Result",
                    lines=8,
                    placeholder="HuggingFace Space preparation results will appear here..."
                )
                
                gr.HTML("""
                <div class="status-card">
                    <h4>📝 Available Space Templates:</h4>
                    <ul>
                        <li>🎤 <strong>Voice Assistant:</strong> Full voice interaction with Whisper → LLaMA-3</li>
                        <li>📄 <strong>Business Tools:</strong> Document processing with Phi-3</li>
                        <li>🎓 <strong>Tutor Bot:</strong> Educational assistant with Mistral-7B</li>
                    </ul>
                </div>
                """)
                
                create_space_btn.click(
                    fn=dashboard.create_hf_space,
                    inputs=[space_name, space_type],
                    outputs=[space_result]
                )
            
            # Deployment Log Tab
            with gr.TabItem("📋 Deployment Log"):
                gr.HTML("<h3>📊 Deployment Activity Log</h3>")
                
                log_refresh_btn = gr.Button("🔄 Refresh Log", variant="secondary")
                deployment_log = gr.Textbox(
                    label="Recent Deployment Activities",
                    lines=20,
                    value=dashboard.get_deployment_log(),
                    max_lines=30
                )
                
                log_refresh_btn.click(
                    fn=dashboard.get_deployment_log,
                    outputs=[deployment_log]
                )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>🌟 ZamAI Pro Models Strategy | 🚀 Production-Ready AI Platform</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
