test:
	python3 -m pytest --pylint -v -s --durations=0 \
		--cov=minimal_animatediff --cov-fail-under=85

.PHONY: test
