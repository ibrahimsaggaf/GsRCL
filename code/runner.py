import config
from metrics import metrics_list
from collector import res_collector
from experiment_builder import Pipeline


res_collector.id_ = config.ID_
res_collector.path = config.PATH_
res_collector.dataset = config.DATASET
res_collector.labels = config.FILE_Y

res_collector.c_losses_list = config.CL_LOSSES
res_collector.cl_aug_methods_list = config.CL_AUG_METHODS
res_collector.cl_epochs = config.CL_EPOCHS
res_collector.cl_step = config.CL_STEP
res_collector.cl_batch_size = config.CL_BATCH_SIZE
res_collector.top_n_genes = config.TOP_N_GENES

res_collector.clasifiers = config.CLASSIFIERS
res_collector.params_grid = config.PARAMS_GRID
res_collector.cv = config.CV
res_collector.train_size = config.TRAIN_SIZE
res_collector.metric = config.METRIC
res_collector.world_size = config.WORLD_SIZE
res_collector.num_cores = config.NUM_CORES


if __name__ == '__main__':
    pipeline = Pipeline(clfs=config.CLASSIFIERS, params_grid=config.PARAMS_GRID, metric=config.METRIC, metrics_list=metrics_list, 
                        c_losses_list=config.CL_LOSSES, cl_aug_methods_list=config.CL_AUG_METHODS, file_X=config.FILE_X, file_y=config.FILE_Y, 
                        dataset=config.DATASET, path=config.PATH_, cv=config.CV, train_size=config.TRAIN_SIZE)
    pipeline.run()
