from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def compute_iou(y_true, y_pred, num_classes):
        iou_scores = []
        for cls in range(num_classes):
            intersection = np.logical_and(y_true[:, cls], y_pred[:, cls]).sum()
            union = np.logical_or(y_true[:, cls], y_pred[:, cls]).sum()
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)
        return np.mean(iou_scores)
    
def compute_metrics(y_true, y_pred, num_classes, ):   
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Вычисление IoU
    iou = compute_iou(y_true, y_pred, num_classes)

    metrics_dict = {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'IoU': iou
    }

    text_metrics = f"Precision (macro): {metrics_dict['Precision']:.4f}\n"
    text_metrics += f"Recall (macro): {metrics_dict['Recall']:.4f}\n"
    text_metrics += f"F1 (macro): {metrics_dict['F1']:.4f}\n"
    text_metrics += f"IoU-score: {metrics_dict['IoU']:.4f}\n"

    return metrics_dict, text_metrics
