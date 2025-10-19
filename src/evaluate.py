import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from models.three_stream_model import ThreeStreamModel
from dataset_multimodal import create_dataloaders

class ModelEvaluator:
    """Comprehensive model evaluation"""
    def __init__(self, model, dataloader, device, class_names):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names
    
    def evaluate(self):
        """Run full evaluation"""
        print("Running evaluation...")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        # Per-stream predictions
        rgb_predictions = []
        pose_predictions = []
        flow_predictions = []
        
        # Gate weights for analysis
        all_gate_weights = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                rgb, pose, angles, flow, labels, video_ids = batch
                
                rgb = rgb.to(self.device)
                pose = pose.to(self.device)
                angles = angles.to(self.device)
                flow = flow.to(self.device)
                
                # Get predictions
                logits, stream_outputs = self.model(
                    rgb, pose, angles, flow,
                    return_stream_outputs=True
                )
                
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Store per-stream predictions
                _, rgb_pred = stream_outputs['rgb'].max(1)
                _, pose_pred = stream_outputs['pose'].max(1)
                _, flow_pred = stream_outputs['flow'].max(1)
                
                rgb_predictions.extend(rgb_pred.cpu().numpy())
                pose_predictions.extend(pose_pred.cpu().numpy())
                flow_predictions.extend(flow_pred.cpu().numpy())
                
                # Store gate weights
                if stream_outputs['gate_weights'] is not None:
                    all_gate_weights.extend(stream_outputs['gate_weights'].cpu().numpy())
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        rgb_predictions = np.array(rgb_predictions)
        pose_predictions = np.array(pose_predictions)
        flow_predictions = np.array(flow_predictions)
        
        all_gate_weights = np.array(all_gate_weights) if all_gate_weights else None
        
        # Compute metrics
        results = self.compute_metrics(
            all_labels,
            all_predictions,
            all_probs,
            rgb_predictions,
            pose_predictions,
            flow_predictions,
            all_gate_weights
        )
        
        return results
    
    def compute_metrics(
        self,
        labels,
        predictions,
        probs,
        rgb_pred,
        pose_pred,
        flow_pred,
        gate_weights
    ):
        """Compute all metrics"""
        results = {}
        
        # Overall accuracy
        results['accuracy'] = accuracy_score(labels, predictions)
        
        # Per-stream accuracy
        results['rgb_accuracy'] = accuracy_score(labels, rgb_pred)
        results['pose_accuracy'] = accuracy_score(labels, pose_pred)
        results['flow_accuracy'] = accuracy_score(labels, flow_pred)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(labels, predictions)
        
        # Classification report
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        results['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        # Macro averages
        results['macro_precision'] = precision.mean()
        results['macro_recall'] = recall.mean()
        results['macro_f1'] = f1.mean()
        
        # Gate weight statistics
        if gate_weights is not None:
            results['gate_weights'] = {
                'mean': gate_weights.mean(axis=0).tolist(),
                'std': gate_weights.std(axis=0).tolist()
            }
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path='results/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix to {save_path}")
    
    def plot_per_class_metrics(self, results, save_path='results/per_class_metrics.png'):
        """Plot per-class metrics"""
        classes = list(results['per_class'].keys())
        metrics = ['precision', 'recall', 'f1']
        
        data = {metric: [] for metric in metrics}
        for class_name in classes:
            for metric in metrics:
                data[metric].append(results['per_class'][class_name][metric])
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, data['precision'], width, label='Precision')
        ax.bar(x, data['recall'], width, label='Recall')
        ax.bar(x + width, data['f1'], width, label='F1-Score')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved per-class metrics to {save_path}")
    
    def print_report(self, results):
        """Print evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
        
        print(f"\nPer-Stream Accuracy:")
        print(f"  RGB Stream:  {results['rgb_accuracy']:.4f}")
        print(f"  Pose Stream: {results['pose_accuracy']:.4f}")
        print(f"  Flow Stream: {results['flow_accuracy']:.4f}")
        
        print(f"\nMacro Averages:")
        print(f"  Precision: {results['macro_precision']:.4f}")
        print(f"  Recall:    {results['macro_recall']:.4f}")
        print(f"  F1-Score:  {results['macro_f1']:.4f}")
        
        print(f"\nPer-Class Performance:")
        for class_name, metrics in results['per_class'].items():
            print(f"\n  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1']:.4f}")
            print(f"    Support:   {metrics['support']}")
        
        if 'gate_weights' in results:
            print(f"\nAverage Gate Weights:")
            weights = results['gate_weights']['mean']
            print(f"  RGB:  {weights[0]:.4f}")
            print(f"  Pose: {weights[1]:.4f}")
            print(f"  Flow: {weights[2]:.4f}")
        
        print("\n" + "="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Three-Stream Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/videos')
    parser.add_argument('--pose_dir', type=str, default='data/poses')
    parser.add_argument('--flow_dir', type=str, default='data/optical_flow')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = ThreeStreamModel(num_classes=6, num_frames=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val accuracy: {checkpoint['val_acc']:.4f}")
    
    # Create dataloader (using val split)
    _, val_loader = create_dataloaders(
        video_dir=args.data_dir,
        pose_dir=args.pose_dir,
        flow_dir=args.flow_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1
    )
    
    # Class names
    class_names = ['locomotion', 'manipulation', 'hygiene', 'eating', 'communication', 'leisure']
    
    # Evaluate
    evaluator = ModelEvaluator(model, val_loader, device, class_names)
    results = evaluator.evaluate()
    
    # Print report
    evaluator.print_report(results)
    
    # Save plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    evaluator.plot_per_class_metrics(
        results,
        save_path=output_dir / 'per_class_metrics.png'
    )
    
    # Save results to JSON
    results_copy = results.copy()
    results_copy['confusion_matrix'] = results_copy['confusion_matrix'].tolist()
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
