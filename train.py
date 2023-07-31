import argparse

import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer
from trainer.seq2seq_trainer import Seq2SeqTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.train_opts(parser)
    opt = parser.parse_args()

    if opt.model_choice == 'transformer':
        trainer = TransformerTrainer(opt)
    elif opt.model_choice == 'seq2seq':
        trainer = Seq2SeqTrainer(opt)
    # trainer.train(opt)

    study = optuna.create_study(direction="maximize")
    study.optimize(trainer.train(opt), n_trials=100, timeout=600)

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
