import torch
import os



def save_model(path_to_weights,
               name_model,
               model_state_dict,
               optimizer_state_dict,
               num_epochs):
    # Сохраняем модель
    path = os.path.join(path_to_weights, f"{name_model}.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict},
        path)


def save_metrics_train(path_to_metrics_train,
                       train_loss_values,
                       valid_loss_values,
                       metric_values,
                       name_metric,
                       date,
                       name_model):
    metrics = {
        'train_loss': train_loss_values,
        'valid_loss': valid_loss_values,
        f'valid_{name_metric}': metric_values
    }
    # Сохранение метрик
    path = os.path.join(path_to_metrics_train,
                        f"train_{name_model}_{date.strftime('%Y-%m-%d_%H-%M-%S')}.pt")
    torch.save(metrics, path)


def save_metrics_test(path_to_metrics_test,
                      name_model,
                      f1,
                      name_metric,
                      date,
                      class_acc_dir=None):
    if class_acc_dir is not None:
        metric = {
            f'{name_metric}_value': f1,
            'Acc_dir': class_acc_dir
        }
    else:
        metric = {
            f'{name_metric}_value': f1
        }
    # Сохранение метрик
    path = os.path.join(path_to_metrics_test,
                        f"test_{name_model}_{date.strftime('%Y-%m-%d_%H-%M-%S')}.pt")
    torch.save(metric, path)
