#!/bin/bash

# ZamAI Production Deployment Script
set -e

echo "🚀 Starting ZamAI Production Deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your tokens"
    exit 1
fi

# Load environment variables
source .env

# Validate required environment variables
required_vars=("HF_TOKEN" "WHISPER_MODEL" "LLAMA3_MODEL" "MISTRAL_EDU_MODEL" "PHI3_BUSINESS_MODEL" "EMBEDDINGS_MODEL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: $var is not set in .env file"
        exit 1
    fi
done

echo "✅ Environment variables validated"

# Build and start services
echo "🏗️ Building Docker containers..."
docker-compose -f docker-compose.production.yml build --no-cache

echo "🚀 Starting ZamAI services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health checks
echo "🏥 Performing health checks..."

check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=10
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        echo "⏳ Waiting for $service_name (attempt $attempt/$max_attempts)..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed health check"
    return 1
}

# Check all services
check_service "API Gateway" 8000
check_service "Voice Assistant" 8001
check_service "Tutor Bot" 8002
check_service "Business Tools" 8003

echo "🎉 All services are healthy!"

# Display service URLs
echo ""
echo "🌐 ZamAI Services are now running:"
echo "├── 🌍 Main Dashboard: http://localhost"
echo "├── 🔌 API Gateway: http://localhost:8000"
echo "├── 🎤 Voice Assistant: http://localhost:8001"
echo "├── 🎓 Tutor Bot: http://localhost:8002"
echo "├── 📄 Business Tools: http://localhost:8003"
echo "└── 📊 Service Status: docker-compose -f docker-compose.production.yml ps"

echo ""
echo "📋 Management Commands:"
echo "├── 📊 View logs: docker-compose -f docker-compose.production.yml logs -f [service]"
echo "├── 🔄 Restart: docker-compose -f docker-compose.production.yml restart [service]"
echo "├── 🛑 Stop: docker-compose -f docker-compose.production.yml down"
echo "└── 🔧 Scale: docker-compose -f docker-compose.production.yml up -d --scale voice-assistant=3"

echo ""
echo "🚀 ZamAI Production Deployment Complete!"
