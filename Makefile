test:
	python3 -m pytest  -v -s \
		--durations=0 \
		--pylint \
		--cov=minimal_animatediff \
		--cov-fail-under=85 \
		--cov-report term \
		--cov-report annotate:coverage

.PHONY: test
