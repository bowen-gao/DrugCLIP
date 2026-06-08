#!/bin/bash
# Quick start script for DrugCLIP Web Application

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      DrugCLIP Web Application - Quick Start                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}[1/5] Initializing project structure...${NC}"
python3 web_app/setup.py

echo ""
echo -e "${BLUE}[2/5] Installing dependencies...${NC}"
cd web_app/backend
pip install -r requirements.txt -q
cd ../..

echo ""
echo -e "${BLUE}[3/5] Checking CUDA availability...${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    print('    ✓ CUDA is available')
    print('    Device:', torch.cuda.get_device_name(0))
else:
    print('    ⚠ CUDA is not available, using CPU')
    print('    Note: CPU mode will be slower. For better performance, install CUDA.')
"

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo ""
echo -e "${BLUE}Option 1: Local Development${NC}"
echo "  1. Download model and data (see README.md)"
echo "  2. Run: ${GREEN}cd web_app/backend && python app.py${NC}"
echo "  3. Open: ${GREEN}http://localhost:5000${NC}"
echo ""
echo -e "${BLUE}Option 2: Docker${NC}"
echo "  1. Make sure Docker and Docker Compose are installed"
echo "  2. Download model and data (see README.md)"
echo "  3. Run: ${GREEN}docker-compose -f web_app/docker-compose.yml up${NC}"
echo "  4. Open: ${GREEN}http://localhost:5000${NC}"
echo ""
echo -e "${YELLOW}📚 Documentation:${NC}"
echo "  - See: web_app/README.md for detailed instructions"
echo "  - API docs in: web_app/README.md (API 文档 section)"
echo ""
echo -e "${YELLOW}🔗 Resources:${NC}"
echo "  - Paper: https://arxiv.org/pdf/2310.06367.pdf"
echo "  - Download data: https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV"
echo "  - PDB Database: https://www.rcsb.org/"
echo ""
