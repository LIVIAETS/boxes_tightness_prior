CC = python3.8
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all plot train pack view metrics report

# CFLAGS = -O
# DEBUG = --debug
EPC = 50
# EPC = 5
BS = 32

K = 2

G_RGX = (\d+_\d+)_\d+_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NET = ENet
# NET = Dummy

TRN = results/atlas/ce \
	results/atlas/box_prior_box_size \
	results/atlas/box_prior_box_size_neg_size \
	results/atlas/box_prior_neg_size \
	results/atlas/ce_residual \
	results/atlas/box_prior_box_size_neg_size_residual \
	results/atlas/box_ce \
	results/atlas/box_ce_thick \
	results/atlas/box_prior_box_size_neg_size_thick

# 	results/atlas/box_prior_neg_size_residual

GRAPH = results/atlas/tra_loss.png results/atlas/val_loss.png \
		results/atlas/tra_dice.png results/atlas/val_dice.png \
		results/atlas/val_3d_dsc.png
# 		results/atlas/val_hausdorff.png \
# HIST =  results/atlas/val_dice_hist.png
HIST =
BOXPLOT = results/atlas/val_dice_boxplot.png \
		results/atlas/val_3d_dsc_boxplot.png
# 		results/atlas/val_hausdorff_boxplot.png \

PLT = $(GRAPH) $(HIST) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-atlas.tar.gz
LIGHTPACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-atlas_light.tar.gz

all: pack

plot: $(PLT)
# plot: results/atlas/val_3d_dsc.png

train: $(TRN)

pack: report $(PACK) $(LIGHTPACK)

$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available
$(LIGHTPACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	$(eval PLTS:=$(filter %.png, $^))
	$(eval FF:=$(filter-out %.png, $^))
	$(eval TGT:=$(addsuffix /best_epoch, $(FF)) $(addsuffix /*.npy, $(FF)) $(addsuffix /best_epoch.txt, $(FF)) $(addsuffix /metrics.csv, $(FF)))
	tar cf - $(PLTS) $(TGT) | pigz > $@
	chmod -w $@


# Dataset
data/atlas: data/atlas.lineage data/ATLAS_R1.1.zip
	sha256sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	mv $@_tmp/ATLAS_R1.1/* $@_tmp
	rmdir $@_tmp/ATLAS_R1.1
	ls $@_tmp | grep Site
	for f in `ls $@_tmp | grep Site` ; do \
		ls -1 $@_tmp/$$f >> $@_tmp/all_ids ; \
		mv $@_tmp/$$f/* $@_tmp ; \
	done
	rmdir $@_tmp/Site*
	echo `wc -l $@_tmp/all_ids` patients total
	sort $@_tmp/all_ids > $@_tmp/sorted_ids
	uniq $@_tmp/sorted_ids > $@_tmp/uniq_ids
	echo `wc -l $@_tmp/uniq_ids` unique patients
	mv $@_tmp $@


data/ATLAS/train/gt data/ATLAS/val/gt: | data/ATLAS
data/ATLAS/train data/ATLAS/val: | data/ATLAS
data/ATLAS: data/atlas
	rm -rf $@_tmp $@
	$(PP) $(CC) $(CFLAGS) preprocess/slice_atlas.py --source_dir $^ --dest_dir $@_tmp --id_list $^/uniq_ids \
		--n_augment 0 --shape 208 256
	mv $@_tmp $@


# Weak labels generation
weaks = data/ATLAS/train/centroid data/ATLAS/val/centroid \
		data/ATLAS/train/erosion data/ATLAS/val/erosion \
		data/ATLAS/train/random data/ATLAS/val/random \
		data/ATLAS/train/box data/ATLAS/val/box \
		data/ATLAS/train/thickbox data/ATLAS/val/thickbox
weak: $(weaks)

data/ATLAS/train/centroid data/ATLAS/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/ATLAS/train/erosion data/ATLAS/val/erosion: OPT = --seed=0 --strategy=erosion_strat --max_iter 9
data/ATLAS/train/random data/ATLAS/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat
data/ATLAS/train/box data/ATLAS/val/box: OPT = --seed=0 --margin 0 --strategy=box_strat --allow_overflow --allow_bigger
data/ATLAS/train/thickbox data/ATLAS/val/thickbox: OPT = --seed=0 --margin=5 --strategy=box_strat --allow_bigger --allow_overflow

$(weaks): | data/ATLAS/train/gt data/ATLAS/val/gt
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp \
		--quiet --per_connected_components $(OPT)
	mv $@_tmp $@



# Trainings
results/atlas/ce: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/atlas/ce: data/ATLAS/train/gt data/ATLAS/val/gt
results/atlas/ce: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/atlas/box_prior_box_size: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.30, 0.75]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/atlas/box_prior_box_size_thinner: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.30, 0.75]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 2}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size_thinner: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size_thinner: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/atlas/box_prior_box_size_thinner_lighter: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-3), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.30, 0.75]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 2}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size_thinner_lighter: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size_thinner_lighter: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/atlas/box_prior_box_size_lighter: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-3), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.30, 0.75]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size_lighter: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size_lighter: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"

results/atlas/box_prior_negative_ce_box_size: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('CrossEntropy', {'idc': [0]}, None, None, None, 1e-1), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.30, 0.75]}, 'soft_size', 1)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_negative_ce_box_size: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_negative_ce_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"


results/atlas/box_prior_box_size_neg_size: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size_neg_size: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size_neg_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"


results/atlas/box_prior_box_size_thinner_lighter_const_compact: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-3), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.30, 0.75]}, 'soft_size', 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'ConstantBounds', {'values': {1: [-10, 5]}}, 'soft_compactness', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 2}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size_thinner_lighter_const_compact: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size_thinner_lighter_const_compact: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"


results/atlas/box_prior_neg_size: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_neg_size: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_neg_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"


results/atlas/box_ce: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/atlas/box_ce: data/ATLAS/train/gt data/ATLAS/val/gt
results/atlas/box_ce: DATA = --folders="$(B_DATA)+[('box', gt_transform, True)]"


# THICK
results/atlas/box_ce_thick: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/atlas/box_ce_thick: data/ATLAS/train/thickbox data/ATLAS/val/thickbox
results/atlas/box_ce_thick: DATA = --folders="$(B_DATA)+[('thickbox', gt_transform, True)]"

results/atlas/box_prior_box_size_neg_size_thick: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size_neg_size_thick: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size_neg_size_thick: DATA = --folders="$(B_DATA)+[('thickbox', gt_transform, True), ('thickbox', gt_transform, True), ('thickbox', gt_transform, True)]"


# Redisual
results/atlas/ce_residual: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/atlas/ce_residual: data/ATLAS/train/gt data/ATLAS/val/gt
results/atlas/ce_residual: NET = ResidualUNet
results/atlas/ce_residual: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/atlas/box_prior_box_size_neg_size_residual: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, -0.1)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'LogBarrierLoss', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_box_size_neg_size_residual: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_box_size_neg_size_residual: NET = ResidualUNet
results/atlas/box_prior_box_size_neg_size_residual: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"

results/atlas/box_prior_neg_size_residual: OPT = --losses="[('BoxPrior', {'idc': [1], 't': 5}, None, None, None, 1e-4), \
	('NegSizeLoss', {'idc': [0], 't': 5}, None, None, None, 1-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPrior', 'NegSizeLoss'], 'mu': 1.1}" --temperature 1
results/atlas/box_prior_neg_size_residual: data/ATLAS/train/box data/ATLAS/val/box
results/atlas/box_prior_neg_size_residual: NET = ResidualUNet
results/atlas/box_prior_neg_size_residual: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"


# Template
results/atlas/%:
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=$(K) --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


results/atlas/deepcut: data/ATLAS/train/box data/ATLAS/val/box
	rm -rf $@_tmp
	$(CC) $(CFLAGS) deepcut.py --dataset $(dir $(<D)) --batch_size $(BS) --schedule \
		--in_memory --n_epoch $(EPC) --network $(NET) --n_class 2 --metric_axis=1 \
		--img_size 208 256 \
		--grp_regex="$(G_RGX)" --workdir $@_tmp --save_train $(DEBUG)
	mv $@_tmp $@


# Metrics
metrics: $(TRN) $(addsuffix /val_3d_dsc.npy, $(TRN)) # $(addsuffix /val_3d_hausdorff.npy, $(TRN))

results/atlas/%/val_3d_dsc.npy: data/ATLAS/val/gt
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics 3d_dsc \
		--grp_regex "$(G_RGX)" --num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $^


# Plotting
results/atlas/tra_dice.png results/atlas/val_dice.png results/atlas/val_3d_dsc.png: COLS = 1
results/atlas/tra_loss.png results/atlas/val_loss.png: COLS = 0 1 2
results/atlas/tra_loss.png results/atlas/val_loss.png: OPT = --dynamic_third_axis
results/atlas/val_3d_dsc.png: | $(addsuffix /val_3d_dsc.npy, $(TRN))

results/atlas/val_3d_hausdorff.png: COLS = 1
results/atlas/val_3d_hausdorff.png: OPT = --ylim 0 50 --min
results/atlas/val_3d_hausdorff.png: | $(addsuffix /val_3d_hausdorff.npy, $(TRN))
$(GRAPH): plot.py $(TRN)

results/atlas/val_dice_boxplot.png results/atlas/val_3d_hausdorff_boxplot.png: COLS = 1
results/atlas/val_3d_dsc_boxplot.png: COLS = 1
results/atlas/val_3d_hausdorff_boxplot.png: OPT = --epc 199 --ylim 0 50
results/atlas/val_dice_boxplot.png: OPT = --epc 199
results/atlas/val_3d_dsc_boxplot.png: OPT = --epc 199
results/atlas/val_3d_dsc_boxplot.png:| $(addsuffix /val_3d_dsc.npy, $(TRN))
results/atlas/val_3d_hausdorff_boxplot.png: | $(addsuffix /val_3d_hausdorff.npy, $(TRN))
$(BOXPLOT): moustache.py $(TRN)

results/atlas/%.png:
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless $(OPT) $(DEBUG)


# Viewing
view: $(TRN) | data/ATLAS weak
	$(CC) $(CFLAGS) viewer/viewer.py -n 1 --img_source data/ATLAS/val/img data/ATLAS/val/gt \
		data/ATLAS/val/box \
		$(addsuffix /best_epoch/val, $^) \
		--display_names gt box $(notdir $^) \
		--no_contour

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_dice val_3d_dsc --axises 1 --precision 3