import argparse

import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer
from trainer.seq2seq_trainer import Seq2SeqTrainer


import optuna
from optuna.trial import TrialState

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.train_opts(parser)
    opt = parser.parse_args()

    if opt.model_choice == 'transformer':
        def objective(trial):
            trainer = TransformerTrainer(opt)
            loss_epoch_train, loss_epoch_validation, accuracy = trainer.train(opt=opt,trial=trial)
            return accuracy
    # elif opt.model_choice == 'seq2seq':
    #     def objective(trial):
    #         trainer = Seq2seqt(opt)
    #         trainer.train(opt=opt,trial=trial)

    study = optuna.create_study(study_name="transformer-original-source2target-optuna-study", direction="maximize")
    study.optimize(objective,n_jobs = -1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))