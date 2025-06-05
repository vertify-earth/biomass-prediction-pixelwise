#!/usr/bin/env python3
"""
Main entry point for biomass prediction pipeline
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from datetime import datetime
from configs.config import BiomassPipelineConfig
from src.training.pipeline import BiomassPredictionPipeline

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Biomass Prediction Pipeline')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'full'],
                       help='Pipeline mode: test (quick) or full (complete)')
    parser.add_argument('--data_dir', type=str, help='Directory containing data files')
    parser.add_argument('--results_dir', type=str, help='Directory to save results')
    parser.add_argument('--deploy', action='store_true', help='Deploy model to HuggingFace after training')
    parser.add_argument('--hf_repo', type=str, help='HuggingFace repository name')
    parser.add_argument('--hf_token', type=str, help='HuggingFace token')
    
    args = parser.parse_args()
    
    # Create configuration
    config = BiomassPipelineConfig(mode=args.mode)
    
    # Override configuration if specified
    if args.data_dir:
        config.data_dir = args.data_dir
    
    if args.results_dir:
        config.results_dir = args.results_dir
    
    if args.hf_repo:
        config.huggingface_repo = args.hf_repo
    
    # Print banner
    print("\n" + "="*70)
    print(f"🌟 BIOMASS PREDICTION PIPELINE - {args.mode.upper()} MODE 🌟")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: {config.created_by}")
    print(f"Data directory: {config.data_dir}")
    print(f"Results directory: {config.results_dir}")
    print("="*70 + "\n")
    
    # Run pipeline
    try:
        pipeline = BiomassPredictionPipeline(config)
        results = pipeline.run()
        
        # Deploy to HuggingFace if requested
        if args.deploy:
            try:
                from deployment.huggingface_deploy import prepare_huggingface_repo, deploy_to_huggingface
                import tempfile
                
                print("\n" + "="*70)
                print("🚀 DEPLOYING MODEL TO HUGGINGFACE")
                print("="*70)
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Prepare repo
                    model_dir = results['model_dir']
                    success = prepare_huggingface_repo(model_dir, tmp_dir)
                    
                    if success:
                        # Deploy to HuggingFace
                        hf_repo = config.huggingface_repo
                        deploy_to_huggingface(tmp_dir, hf_repo, args.hf_token)
                        
                        print(f"✅ Model deployed to: https://huggingface.co/spaces/{hf_repo}")
                    else:
                        print("❌ Failed to prepare HuggingFace deployment")
                
            except ImportError:
                print("❌ huggingface_hub package not installed. Please install with: pip install huggingface_hub")
            except Exception as e:
                print(f"❌ Error deploying to HuggingFace: {e}")
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        print("\n" + "="*70)
        print("❌ PIPELINE FAILED")
        print(f"Error: {e}")
        print("="*70)
        
        return 1

if __name__ == "__main__":
    sys.exit(main())