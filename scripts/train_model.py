from model_setup import model, train_data, val_data

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # Adjust the number of epochs as needed
    verbose=1
)

# Save the trained model
model.save('helmet_detection_model.h5')
