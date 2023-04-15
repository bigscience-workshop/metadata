check_dirs := bsmetadata

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# this target runs checks on all files and potentially modifies some of them
style:
	black $(check_dirs)
	isort $(check_dirs)
