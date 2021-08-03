# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := .

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	python utils/custom_init_isort.py --check_only
	flake8 $(check_dirs)

# this target runs checks on all files and potentially modifies some of them
style:
	black $(check_dirs)
	isort $(check_dirs)
	