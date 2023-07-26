format:
	python3 -m black minimal_animatediff scripts tests \
		--line-length 120

lint: format
	python3 -m flake8 minimal_animatediff scripts tests \
		--max-line-length 120

test: lint
	python3 -m pytest  -v -s \
		--durations=0 \
		--cov=minimal_animatediff \
		--cov-fail-under=85 \
		--cov-report term \
		--cov-report annotate:coverage

.PHONY: format lint test
