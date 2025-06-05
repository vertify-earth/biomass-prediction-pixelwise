import os
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import json
import sys
from tqdm import tqdm

from configs.config import BiomassPipelineConfig
from src.models.model import initialize_model
from src.data.data_processing import prepare_training_data
from src.training.train import BiomassModelTrainer

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiomassPredictionPipeline:
    """End-to-end pipeline for biomass prediction"""
    
    def __init__(self, config=None):
        """Initialize the pipeline with configuration"""
        self.config = config if config else BiomassPipelineConfig()
        self._setup_logging()
        
        logger.info(f"Biomass Prediction Pipeline initialized in {self.config.mode} mode")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.config.results_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{self.config.mode}_{timestamp}.log")
        
        # Add file handler to root logger
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    def run(self):
        """Execute the complete pipeline"""
        start_time = time.time()
        logger.info("Starting biomass prediction pipeline execution")
        
        # Step 1: Data preparation
        logger.info("Step 1: Data Processing")
        data_dict = prepare_training_data(self.config)
        
        # Step 2: Model initialization
        logger.info("Step 2: Model Initialization")
        model = initialize_model(self.config, data_dict['n_features'])
        logger.info(f"Initialized {self.config.model_type} with {data_dict['n_features']} input features")
        
        # Step 3: Model training
        logger.info("Step 3: Model Training")
        trainer = BiomassModelTrainer(self.config)
        trained_model = trainer.train(model, data_dict['train_loader'], data_dict['val_loader'])
        
        # Step 4: Model evaluation
        logger.info("Step 4: Model Evaluation")
        original_scale = self.config.use_log_transform
        evaluation_results = trainer.evaluate(
            trained_model, 
            data_dict['test_loader'],
            original_scale=original_scale,
            epsilon=self.config.epsilon
        )
        
        # Step 5: Save model and results
        logger.info("Step 5: Saving Results")
        results_dir = os.path.join(self.config.results_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save training history plot
        history_plot_path = os.path.join(results_dir, 'training_history.png')
        trainer.plot_training_history(save_path=history_plot_path)
        
        # Save predictions plot
        predictions_plot_path = os.path.join(results_dir, 'predictions.png')
        trainer.plot_predictions(
            evaluation_results,
            title=f"{self.config.model_type}",
            save_path=predictions_plot_path
        )
        
        # Save model package
        model_dir = trainer.save_model(trained_model, data_dict, evaluation_results)
        
        # Create pipeline metadata
        pipeline_metadata = {
            'execution_time': time.time() - start_time,
            'completion_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'mode': self.config.mode,
            'model_type': self.config.model_type,
            'model_dir': model_dir,
            'n_features': data_dict['n_features'],
            'created_by': self.config.created_by,
            'created_date': self.config.created_date
        }
        
        # Add evaluation metrics
        if 'original_scale' in evaluation_results:
            pipeline_metadata['log_r2'] = float(evaluation_results['log_scale']['metrics']['r2'])
            pipeline_metadata['original_r2'] = float(evaluation_results['original_scale']['metrics']['r2'])
        else:
            pipeline_metadata['r2'] = float(evaluation_results['metrics']['r2'])
        
        # Save metadata
        metadata_path = os.path.join(results_dir, 'pipeline_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(pipeline_metadata, f, indent=2)
        
        logger.info(f"Pipeline completed in {pipeline_metadata['execution_time']:.2f} seconds")
        logger.info(f"Results saved to {self.config.results_dir}")
        
        return {
            'model': trained_model,
            'model_dir': model_dir,
            'evaluation': evaluation_results,
            'data': data_dict,
            'pipeline_metadata': pipeline_metadata
        }

def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Biomass Prediction Pipeline')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'full'],
                       help='Pipeline mode: test (quick) or full (complete)')
    args = parser.parse_args()
    
    # Create configuration
    config = BiomassPipelineConfig(mode=args.mode)
    
    # Run pipeline
    pipeline = BiomassPredictionPipeline(config)
    results = pipeline.run()
    
    # Print summary
    log_r2 = results['pipeline_metadata'].get('log_r2', None)
    orig_r2 = results['pipeline_metadata'].get('original_r2', None)
    r2 = results['pipeline_metadata'].get('r2', None)
    
    print("\n" + "="*60)
    print(f"🌟 BIOMASS PREDICTION PIPELINE COMPLETED 🌟")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {config.mode.upper()}")
    print(f"Model: {config.model_type}")
    
    if orig_r2 is not None:
        print(f"R² (Original Scale): {orig_r2:.4f}")
        print(f"R² (Log Scale): {log_r2:.4f}")
    else:
        print(f"R²: {r2:.4f}")
    
    print(f"Results saved to: {config.results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()