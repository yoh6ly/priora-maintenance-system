"""
Batch processing example for PRIORA System
"""

import pandas as pd
from src.priora import PrioraSystem

print("🚀 Initializing PRIORA System for Batch Processing...")
priora = PrioraSystem()

# Create sample data
print("\n📊 Creating sample motor dataset...")
sample_data = pd.DataFrame({
    'voltage': [380, 380, 230, 380, 220, 380],
    'current': [12.5, 18.2, 7.8, 14.3, 9.5, 16.8],
    'power': [6500, 9200, 1600, 7500, 2000, 8800],
    'temperature': [62, 78, 58, 71, 65, 76],
    'vibration': [3.2, 6.5, 2.8, 4.1, 3.5, 5.9]
})

print(f"\nSample Data ({len(sample_data)} motors):")
print(sample_data)

# Batch prediction
print("\n⚙️ Processing batch predictions...")
predictions = priora.predict_batch(sample_data)

# Display results
print("\n✅ Prediction Results:")
print(predictions[['motor_type', 'fault_probability', 'priority']])

# Generate summary
print("\n📈 Summary Statistics:")
summary = priora.get_summary(predictions)
print(f"  Total Motors: {summary['total_motors']}")
print(f"  Average Fault Probability: {summary['average_fault_probability']:.2f}%")
print(f"  Critical Motors: {summary['critical_motors']} ({summary['critical_percentage']:.1f}%)")
print(f"  Motor Types Distribution: {summary['motor_types']}")
print(f"  Priority Distribution: {summary['priority_distribution']}")

# Export results
output_file = 'results/predictions/batch_predictions.csv'
priora.export_results(predictions, output_file)
print(f"\n💾 Results exported to: {output_file}")

print("\n✅ Batch processing completed!")
