VENV = .venv
UV = uv

.PHONY: install sync run clean

# Create virtual environment only if missing
$(VENV):
	$(UV) venv $(VENV)

# Install dependencies
install: $(VENV)
	$(UV) pip install -e .

# Sync dependencies from lockfile
sync: $(VENV)
	$(UV) sync

# Run the application
run:
	$(UV) run uvicorn main:app --reload

# Remove virtual environment
clean:
	rm -rf $(VENV)