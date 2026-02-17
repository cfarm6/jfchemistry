if [ ! -d "JoltQC" ]; then
    echo "JoltQC directory not found. Cloning from GitHub..."
    git clone https://github.com/ByteDance-Seed/JoltQC.git
fi

pip install -e JoltQC
