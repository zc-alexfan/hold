.PHONY: clean fix imports sort pull

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf condor_logs/*
	find logs/* -delete
fix:
	black src *.py
sort:
	isort src *.py --wrap-length=1 --combine-as --trailing-comma --use-parentheses
imports:
	autoflake -i -r --remove-all-unused-imports src *.py
pull:
	rsync -ravz --no-g --no-p --progress  zfan@129.132.75.187:/home/zfan/vid2avatar/code/data /is/cluster/fast/fzicong/blaze/
	rsync -ravz --no-g --no-p --progress  zfan@129.132.75.187:/home/zfan//vid2avatar/code/saved_models /is/cluster/fast/fzicong/blaze/
