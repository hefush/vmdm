# Contributing to VMDM

## Bug Reports

Please report bugs via [GitHub Issues](https://github.com/hefush/vmdm/issues). Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)

## Code Contributions

1. Fork the repository
2. Create a feature branch
3. Make changes following PEP 8 style guidelines
4. Test thoroughly
5. Submit pull request with clear description

## Development Setup

```bash
git clone https://github.com/hefush/vmdm.git
cd vmdm
mamba env create -f requirements.yaml -p ./venv
conda activate ./venv
```

If Mamba is not available, use `conda env create -f requirements.yaml -p ./venv`.
