CC = python3.8
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all plot train pack view metrics report

# CFLAGS = -O
# DEBUG = --debug
EPC = 100
# EPC = 5
K = 2

BS = 4


G_RGX = (\d+_Case\d+_\d+)_\d+
NET = ResidualUNet
SAVE = --save_train
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]

# TRN = results/prostate/fs \
# 	results/prostate/box_prior_box_size \
# 	results/prostate/box_prior_box_size_neg_ce \
# 	results/prostate/box_prior_box_size_neg_size \
# 	results/prostate/box_prior_neg_size \
# 	results/prostate/box_prior_neg_ce \
# 	results/prostate/box_ce

TRN = results/prostate/fs \
	results/prostate/box_prior_box_size_neg_size \
	results/prostate/box_ce \
	results/prostate/box_prior_box_size_neg_size_thick_wider \
	results/prostate/box_ce_thick \
	results/prostate/box_prior_box_size_neg_size_noaug \
	results/prostate/box_ce_noaug \
	results/prostate/box_prior_box_size \
	results/prostate/box_prior_box_size_neg_ce \
	results/prostate/deepcut
# 	 \
# 	results/prostate/deepcut2

# 	results/prostate/box_prior_box_size_neg_size_thick \
# 	results/prostate/box_prior_box_size_neg_size \


GRAPH = results/prostate/val_dice.png results/prostate/tra_dice.png \
		results/prostate/val_loss.png results/prostate/tra_loss.png \
		results/prostate/val_3d_dsc.png
# 		results/prostate/val_3d_hausdorff.png
HIST =
BOXPLOT = results/prostate/val_3d_dsc_boxplot.png results/prostate/val_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)


REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-prostate.tar.gz

all: pack

plot: $(PLT)

train: $(TRN)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	# tar -zc -f $@ $^  # Use if pigz is not available
	tar cf - $^ | pigz > $@
	chmod -w $@


# Extraction and slicing
data/PROSTATE/train/gt data/PROSTATE/val/gt: data/PROSTATE
data/PROSTATE: data/promise
	rm -rf $@_tmp
	$(PP) $(CC) $(CFLAGS) preprocess/slice_promise.py --source_dir $< --dest_dir $@_tmp --n_augment=0
	mv $@_tmp $@
data/promise: data/prostate.lineage data/TrainingData_Part1.zip data/TrainingData_Part2.zip data/TrainingData_Part3.zip
	md5sum -c $<
	rm -rf $@_tmp
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	unzip -q $(word 4, $^) -d $@_tmp
	mv $@_tmp $@


# Weak labels generation
weaks = data/PROSTATE/train/centroid data/PROSTATE/val/centroid \
		data/PROSTATE/train/erosion data/PROSTATE/val/erosion \
		data/PROSTATE/train/random data/PROSTATE/val/random \
		data/PROSTATE/train/box data/PROSTATE/val/box \
		data/PROSTATE/train/thickbox data/PROSTATE/val/thickbox


weak: $(weaks)

data/PROSTATE/train/centroid data/PROSTATE/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/PROSTATE/train/erosion data/PROSTATE/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/PROSTATE/train/random data/PROSTATE/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat
data/PROSTATE/train/box data/PROSTATE/val/box: OPT = --seed=0 --margin=0 --strategy=box_strat --allow_bigger --allow_overflow
data/PROSTATE/train/thickbox data/PROSTATE/val/thickbox: OPT = --seed=0 --margin=10 --strategy=box_strat --allow_bigger --allow_overflow

$(weaks): data/PROSTATE
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp $(OPT)
	mv $@_tmp $@


data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box: | data/PROSTATE-Aug
data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt: data/PROSTATE-Aug
data/PROSTATE-Aug: data/PROSTATE | weak
	rm -rf $@ $@_tmp
	$(CC) $(CFLAGS) augment.py --n_aug 4 --root_dir $</train --dest_dir $@_tmp/train
	$(CC) $(CFLAGS) augment.py --n_aug 0 --root_dir $</val --dest_dir $@_tmp/val  # Naming scheme for consistency
	mv $@_tmp $@


results/prostate/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/prostate/fs: data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt
results/prostate/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"


results/prostate/box_prior_box_size: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_box_size_wider: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 10}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_wider: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_wider: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_box_size_thinner: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 2}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_thinner: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_thinner: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_box_size_thinnest: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 1}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_thinnest: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_thinnest: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_box_size_lighter: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_lighter: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_lighter: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_box_size_thinner_lighter: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 2}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_thinner_lighter: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_thinner_lighter: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_box_size_lightest: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-6), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_lightest: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_lightest: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_box_size_neg_ce: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2), \
	('CrossEntropy', {'idc': [0]}, None, None, None, 1)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_neg_ce: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_neg_ce: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_heavier_box_size_neg_ce: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('CrossEntropy', {'idc': [0]}, None, None, None, 1)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_heavier_box_size_neg_ce: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_heavier_box_size_neg_ce: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"


results/prostate/box_prior_box_size_neg_size: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_neg_size: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_box_size_neg_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"


results/prostate/box_prior_neg_size: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 5e-3), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_neg_size: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_neg_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_prior_neg_ce: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('CrossEntropy', {'idc': [0]}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_neg_ce: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/box_prior_neg_ce: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_ce: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/prostate/box_ce: data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt
results/prostate/box_ce: DATA = --folders="$(B_DATA)+[('box', gt_transform, True)]"



results/prostate/fs_noaug: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/prostate/fs_noaug: data/PROSTATE/train/gt data/PROSTATE/val/gt
results/prostate/fs_noaug: G_RGX = (Case\d+_\d+)_\d+
results/prostate/fs_noaug: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/prostate/box_ce_noaug: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/prostate/box_ce_noaug: data/PROSTATE/train/box data/PROSTATE/val/box
results/prostate/box_ce_noaug: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_ce_noaug: DATA = --folders="$(B_DATA)+[('box', gt_transform, True)]"

results/prostate/box_prior_box_size_neg_size_noaug: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_neg_size_noaug: data/PROSTATE/train/box data/PROSTATE/val/box
results/prostate/box_prior_box_size_neg_size_noaug: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_prior_box_size_neg_size_noaug: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"

results/prostate/box_ce_thick: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/prostate/box_ce_thick: data/PROSTATE/train/thickbox data/PROSTATE/val/thickbox
results/prostate/box_ce_thick: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_ce_thick: DATA = --folders="$(B_DATA)+[('thickbox', gt_transform, True)]"

results/prostate/box_prior_box_size_neg_size_thick: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_neg_size_thick: data/PROSTATE/train/thickbox data/PROSTATE/val/thickbox
results/prostate/box_prior_box_size_neg_size_thick: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_prior_box_size_neg_size_thick: DATA = --folders="$(B_DATA)+[('thickbox', gt_transform, True), ('thickbox', gt_transform, True), ('thickbox', gt_transform, True)]"


results/prostate/box_prior_box_size_neg_size_thick_wider: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-3), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 20}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/prostate/box_prior_box_size_neg_size_thick_wider: data/PROSTATE/train/thickbox data/PROSTATE/val/thickbox
results/prostate/box_prior_box_size_neg_size_thick_wider: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_prior_box_size_neg_size_thick_wider: DATA = --folders="$(B_DATA)+[('thickbox', gt_transform, True), ('thickbox', gt_transform, True), ('thickbox', gt_transform, True)]"




results/prostate/%:
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--in_memory --compute_3d_dice $(SAVE) \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


results/prostate/deepcut: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
	rm -rf $@_tmp
	$(CC) $(CFLAGS) deepcut.py --dataset $(dir $(<D)) --batch_size $(BS) --schedule \
		--in_memory --n_epoch $(EPC) --network $(NET) --n_class 2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --workdir $@_tmp --save_train $(DEBUG)
	mv $@_tmp $@


# Metrics
metrics: $(TRN) $(addsuffix /val_3d_hausdorff.npy, $(TRN))

results/prostate/box_ce_noaug/val_3d_hausdorff.npy: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_ce_noaug/val_3d_hausdorff.npy: DATA = data/PROSTATE/val/gt

results/prostate/box_ce_thick/val_3d_hausdorff.npy: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_ce_thick/val_3d_hausdorff.npy: DATA = data/PROSTATE/val/gt

results/prostate/box_prior_box_size_neg_size_noaug/val_3d_hausdorff.npy: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_prior_box_size_neg_size_noaug/val_3d_hausdorff.npy: DATA = data/PROSTATE/val/gt

results/prostate/box_prior_box_size_neg_size_thick/val_3d_hausdorff.npy: G_RGX = (Case\d+_\d+)_\d+
results/prostate/box_prior_box_size_neg_size_thick/val_3d_hausdorff.npy: DATA = data/PROSTATE/val/gt


results/prostate/%/val_3d_hausdorff.npy: DATA = data/PROSTATE-Aug/val/gt
results/prostate/%/val_3d_hausdorff.npy:
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics 3d_hausdorff \
		--grp_regex "$(G_RGX)" --num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $(DATA)



# Plotting
results/prostate/val_3d_dsc.png results/prostate/val_dice.png results/prostate/tra_dice.png: COLS = 1
results/prostate/val_dice.png results/prostate/val_3d_dsc.png: plot.py $(TRN)
results/prostate/tra_dice.png: plot.py $(TRN)

results/prostate/tra_loss.png results/prostate/val_loss.png: COLS = 0 1
results/prostate/tra_loss.png results/prostate/val_loss.png: OPT = --ylim 0 10 --dynamic_third_axis
results/prostate/tra_loss.png results/prostate/val_loss.png: plot.py $(TRN)

results/prostate/val_3d_hausdorff.png: COLS = 1
results/prostate/val_3d_hausdorff.png: OPT = --ylim 0 50 --min
results/prostate/val_3d_hausdorff.png: plot.py $(TRN) | metrics

results/prostate/val_3d_dsc_boxplot.png results/prostate/val_dice_boxplot.png: COLS = 1
results/prostate/val_3d_dsc_boxplot.png results/prostate/val_dice_boxplot.png: moustache.py $(TRN)

# Nice titles:
results/prostate/val_3d_dsc.png: OPT = --title "Validation dice over time"
results/prostate/tra_dice.png: OPT = --title "Training dice over time"

$(GRAPH) $(HIST) $(BOXPLOT):
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT) $(DEBUG)

# Viewing
view: $(TRN)
	viewer/viewer.py -n 3 --img_source data/PROSTATE/val/img data/PROSTATE/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^) --no_contour $(DEBUG)

view_epc: $(TRN)
	viewer/viewer.py -n 3 --img_source data/PROSTATE/val/img data/PROSTATE/val/gt $(addsuffix /iter$(ITER)/val, $^) --crop 10 \
		--display_names gt $(notdir $^) $(DEBUG)

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_3d_dsc val_dice --axises 1 --precision 3 \