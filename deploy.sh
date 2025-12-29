#!/bin/bash
set -e

echo "================================"
echo "XGenerator EC2 Deployment Script"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
GITHUB_REPO="https://github.com/AndersonTsaiTW/XGenerator.git"
APP_DIR="$HOME/apps/XGenerator"

echo -e "${YELLOW}Step 1: Installing dependencies...${NC}"
sudo dnf update -y
sudo dnf install -y git docker

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

echo -e "${YELLOW}Step 2: Starting Docker service...${NC}"
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

echo -e "${GREEN}✓ Docker service configured${NC}"
echo ""

echo -e "${YELLOW}Step 3: Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
else
    echo -e "${GREEN}✓ Docker Compose already installed${NC}"
fi

docker-compose --version
echo ""

echo -e "${YELLOW}Step 4: Cloning repository...${NC}"
mkdir -p ~/apps
cd ~/apps

if [ -d "$APP_DIR" ]; then
    echo "Directory exists. Pulling latest changes..."
    cd "$APP_DIR"
    git pull origin main
else
    git clone "$GITHUB_REPO"
    cd XGenerator
fi

echo -e "${GREEN}✓ Repository ready${NC}"
echo ""

echo -e "${YELLOW}Step 5: Setting up environment...${NC}"
if [ ! -f .env ]; then
    echo -e "${RED}⚠ .env file not found!${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo -e "${YELLOW}================================================${NC}"
    echo -e "${YELLOW}IMPORTANT: Please edit .env and add your secrets${NC}"
    echo -e "${YELLOW}================================================${NC}"
    echo ""
    echo "Run: nano .env"
    echo "Add your OPENAI_API_KEY and save"
    echo ""
    echo -e "${RED}Deployment paused. After editing .env, run:${NC}"
    echo -e "${RED}cd $APP_DIR && docker-compose up -d --build${NC}"
    exit 1
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi
echo ""

echo -e "${YELLOW}Step 6: Creating data directories...${NC}"
mkdir -p data/{datasets,artifacts,metadata,users}
mkdir -p data/metadata/{models,schemas}
echo -e "${GREEN}✓ Data directories created${NC}"
echo ""

echo -e "${YELLOW}Step 7: Building and starting Docker containers...${NC}"
# Need to use newgrp or restart session for docker group to take effect
# So we use sudo for the first deployment
sudo docker-compose up -d --build

echo ""
echo -e "${GREEN}✓ Containers started${NC}"
echo ""

echo -e "${YELLOW}Step 8: Checking container status...${NC}"
sleep 3
sudo docker-compose ps
echo ""

echo -e "${YELLOW}Step 9: Viewing logs...${NC}"
echo "Press Ctrl+C to exit log view"
sleep 2
sudo docker-compose logs --tail=50 -f api

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Useful commands:"
echo "  View logs:        cd $APP_DIR && docker-compose logs -f"
echo "  Restart services: cd $APP_DIR && docker-compose restart"
echo "  Stop services:    cd $APP_DIR && docker-compose down"
echo "  Update code:      cd $APP_DIR && git pull && docker-compose up -d --build"
echo ""
