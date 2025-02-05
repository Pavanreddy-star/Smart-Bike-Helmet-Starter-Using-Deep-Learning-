from tensorflow.keras.optimizers import Adam
from model_setup import model, train_data, val_data

# Unfreeze base model layers
model.layers[0].trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fine-tune
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=train_data.samples // 32,
    validation_steps=val_data.samples // 32
)

# Save the fine-tuned model
model.save('models/helmet_detection_finetuned.h5')
print("Fine-tuning complete!")
