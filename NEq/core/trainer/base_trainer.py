from copy import deepcopy
import os
import torch
import torch.nn as nn
import wandb
from collections import defaultdict

from ..utils import dist
from ..utils.config import config
from ..utils.logging import logger
from ..utils.sparse_update_tools import (
    parsed_backward_config,
    manually_initialize_grad_mask,
)
from ..utils.neuron_selections_methods import (
    compute_random_budget_mask,
    compute_full_update,
)
from ..utils.hooks import (
    activate_hooks,
    get_global_gradient_mask,
)
from general_utils import (
    log_masks,
    compute_update_budget,
    count_net_num_conv_params,
)


__all__ = ["BaseTrainer"]


class BaseTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        data_loader,
        criterion,
        optimizer,
        lr_scheduler,
        hooks,
        grad_mask,
    ):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion

        self.best_test = 0.0
        self.best_val = 0.0  # Add best_val variable
        self.start_epoch = 0

        # optimization-related
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # NEq related
        self.hooks = hooks
        self.grad_mask = grad_mask

    @property
    def checkpoint_path(self):
        return os.path.join(config.run_dir, "checkpoint")

    def save(self, epoch=0, is_best=False):
        if dist.rank() == 0:
            checkpoint = {
                "state_dict": self.model.module.state_dict()
                if isinstance(self.model, nn.parallel.DistributedDataParallel)
                else self.model.state_dict(),
                "epoch": epoch,
                "best_test": self.best_test,
                "best_val": self.best_val,  # Add best_val to the checkpoint
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }

            os.makedirs(self.checkpoint_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.checkpoint_path, "ckpt.pth"))

            if is_best:
                torch.save(
                    checkpoint, os.path.join(self.checkpoint_path, "ckpt.best.pth")
                )

    def resume(self):
        model_fname = os.path.join(self.checkpoint_path, "ckpt.pth")
        if os.path.exists(model_fname):
            checkpoint = torch.load(model_fname, map_location="cpu")

            # load checkpoint
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
                logger.info("loaded epoch: %d" % checkpoint["epoch"])
            else:
                logger.info("!!! epoch not found in checkpoint")
            if "best_test" in checkpoint:
                self.best_test = checkpoint["best_test"]
                logger.info("loaded best_test: %f" % checkpoint["best_test"])
            else:
                logger.info("!!! best_test not found in checkpoint")
            if "best_val" in checkpoint:
                self.best_val = checkpoint["best_val"]  # Add loading best_val
                logger.info("loaded best_val: %f" % checkpoint["best_val"])
            else:
                logger.info("!!! best_val not found in checkpoint")
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("loaded optimizer")
            else:
                logger.info("!!! optimizer not found in checkpoint")
            if "lr_scheduler" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logger.info("loaded lr_scheduler")
            else:
                logger.info("!!! lr_scheduler not found in checkpoint")
        else:
            logger.info("Skipping resume... checkpoint not found")


    def load_best_validation_model(self):
        if dist.rank() == 0:
            best_checkpoint_path = os.path.join(self.checkpoint_path, "ckpt.best.pth")
            if os.path.exists(best_checkpoint_path):
                checkpoint = torch.load(best_checkpoint_path, map_location="cuda")

                # load best validation checkpoint
                if hasattr(self.model, "module"):
                    self.model.module.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint["state_dict"])

                # update best validation accuracy
                if "best_val" in checkpoint:
                    self.best_val = checkpoint["best_val"]
                    logger.info("loaded best_val: %f" % checkpoint["best_val"])
                else:
                    logger.info("!!! best_val not found in checkpoint")
            else:
                logger.info("Best validation checkpoint not found")

    def validate(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        raise NotImplementedError

    def run_training(self, total_neurons, total_conv_flops):
        test_info_dict = None
        val_info_dict = None # Add validation information dict
        
        # compute total_num_params of the model
        # total_num_params = count_net_num_conv_params(self.model)

        # # Computing the budget in terms of number of parameters
        # config.NEq_config.glob_num_params = compute_update_budget(
        #     total_num_params, config.NEq_config.ratio
        # )
        # config.NEq_config.glob_num_params = compute_update_budget(
        #     config.NEq_config.total_num_params, config.NEq_config.ratio
        # )

        for epoch in range(
            self.start_epoch,
            config.run_config.n_epochs + config.run_config.warmup_epochs,
        ):
            if epoch == 0:
                log_num_saved_params = {}

                # Selecting neurons to update for first epoch from SU config
                if config.NEq_config.initialization == "SU":
                    print("\n----------------SU-init---------------------\n")
                    config.backward_config = parsed_backward_config(
                        config.backward_config, self.model
                    )
                    manually_initialize_grad_mask(
                        self.hooks,
                        self.grad_mask,
                        self.model,
                        config.backward_config,
                        log_num_saved_params,
                    )

                # Randomly selecting neurons to update for first epoch
                elif "random" in config.NEq_config.initialization:
                    print("\n----------------random-init---------------------\n")
                    hooks_num_params_list = []
                    for k in self.hooks:
                        hooks_num_params_list.append(
                            torch.Tensor(
                                [self.hooks[k].single_neuron_num_params]
                                * self.hooks[k].module.out_channels
                            )
                        )
                    compute_random_budget_mask(
                        self.hooks,
                        self.grad_mask,
                        hooks_num_params_list,
                        log_num_saved_params,
                    )

                # Updating all the neurons for first epoch
                elif "full" in config.NEq_config.initialization:
                    print("\n----------------full-init---------------------\n")
                    hooks_num_params_list = []
                    for k in self.hooks:
                        hooks_num_params_list.append(
                            torch.Tensor(
                                [self.hooks[k].single_neuron_num_params]
                                * self.hooks[k].module.out_channels
                            )
                        )
                    compute_full_update(
                        self.hooks, self.grad_mask, hooks_num_params_list, log_num_saved_params
                    )

            # Log the amount of frozen neurons
            frozen_neurons, saved_flops = log_masks(
                self.model, self.hooks, self.grad_mask, total_neurons, total_conv_flops
            )

            # Train step
            activate_hooks(self.hooks, False)
            train_info_dict = self.train_one_epoch(epoch)
            logger.info(f"epoch {epoch}: f{train_info_dict}")

            # Step to compute neurons velocities
            activate_hooks(self.hooks, True)
            _ = self.validate("vel")

            # Validation step
            activate_hooks(self.hooks, False)

            if (
                (epoch + 1) % config.run_config.eval_per_epochs == 0
                or epoch
                == config.run_config.n_epochs + config.run_config.warmup_epochs - 1
            ):
                activate_hooks(self.hooks, False)
                val_info_dict = self.validate("val")
                is_best = val_info_dict["val/top1"] > self.best_val
                self.best_val = max(val_info_dict["val/top1"], self.best_val)
                if is_best:
                    logger.info(
                        " * New best acc (epoch {}): {:.2f}".format(
                            epoch, self.best_val
                        )
                    )
                val_info_dict["val/best"] = self.best_val
                logger.info(f"epoch {epoch}: {val_info_dict}")

                # save model
                self.save(
                    epoch=epoch,
                    is_best=is_best,
                )


            # # Testing step to observe accuracy evolution
            # if (
            #     (epoch + 1) % config.run_config.eval_per_epochs == 0
            #     or epoch
            #     == config.run_config.n_epochs + config.run_config.warmup_epochs - 1
            # ):
            #     activate_hooks(self.hooks, False)
            #     test_info_dict = self.validate("test")
            #     is_best = test_info_dict["test/top1"] > self.best_test
            #     self.best_test = max(test_info_dict["test/top1"], self.best_test)
            #     if is_best:
            #         logger.info(
            #             " * New best acc (epoch {}): {:.2f}".format(
            #                 epoch, self.best_test
            #             )
            #         )
            #     test_info_dict["test/best"] = self.best_test
            #     logger.info(f"epoch {epoch}: {test_info_dict}")

            #     # save model
            #     self.save(
            #         epoch=epoch,
            #         is_best=is_best,
            #     )

            # Testing step at the last epoch load the best validation model
            if epoch == config.run_config.n_epochs + config.run_config.warmup_epochs - 1:
                activate_hooks(self.hooks, False)
                test_info_dict = self.validate("test")
                is_best = test_info_dict["test/top1"] > self.best_test
                self.best_test = max(test_info_dict["test/top1"], self.best_test)
                if is_best:
                    logger.info(
                        " * New best acc (epoch {}): {:.2f}".format(
                            epoch, self.best_test
                        )
                    )
                test_info_dict["test/best"] = self.best_test
                logger.info(f"epoch {epoch}: {test_info_dict}")

            # Logs
            if dist.rank() <= 0:
                wandb.log(
                    {
                        # "Perc of frozen conv neurons": frozen_neurons,
                        # "FLOPS stats": saved_flops,
                        "train": train_info_dict,
                        "val": val_info_dict, #add val to wandb log
                        "test": test_info_dict,
                        "epochs": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "Saved parameters": log_num_saved_params,
                    }
                )

            # Not reseting grad mask and log in case of SU selection
            if not config.NEq_config.neuron_selection == "SU":
                self.grad_mask = {}
                log_num_saved_params = {}
            elif epoch == 0 and not config.NEq_config.initialization == "SU":
                self.grad_mask = {}
                log_num_saved_params = {}
                config.backward_config = parsed_backward_config(
                    config.backward_config, self.model
                )
                manually_initialize_grad_mask(
                    self.hooks,
                    self.grad_mask,
                    self.model,
                    config.backward_config,
                    log_num_saved_params,
                )

            # Computing the gradients mask on the whole network depending on the neuron selection method
            get_global_gradient_mask(
                log_num_saved_params, self.hooks, self.grad_mask, epoch
            )

        return val_info_dict
