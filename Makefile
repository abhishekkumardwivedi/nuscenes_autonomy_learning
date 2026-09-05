.PHONY: setup dashboard validate smoke download-mini

setup:
	./scripts/setup_runpod.sh

dashboard:
	./scripts/run_dashboard.sh

validate:
	python scripts/validate_repo.py
	./scripts/check_dashboard.sh

smoke:
	python smoke_test.py

download-mini:
	./download_nuscenes_mini.sh $${NUSCENES_DATAROOT:-/workspace/data/nuscenes}
