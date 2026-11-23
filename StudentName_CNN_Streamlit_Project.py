import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10


# ==============================================================================
# PART A: CNN IMAGE CLASSIFICATION (CIFAR-10)
# ==============================================================================

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]


def load_and_explore_data():
    """Step 1 & 2: Load CIFAR-10 dataset, inspect shapes, and preprocess images."""
    print("\n--- STEP 1: LOADING & EXPLORING CIFAR-10 DATASET ---")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(f"Training images shape : {x_train.shape} (Samples, Height, Width, Channels)")
    print(f"Training labels shape : {y_train.shape}")
    print(f"Testing images shape  : {x_test.shape}")
    print(f"Testing labels shape  : {y_test.shape}")
    print(f"Number of classes     : {len(CLASS_NAMES)} -> {CLASS_NAMES}")

    # Step 2: Preprocess images (normalize pixel values from [0, 255] to [0, 1])
    print("\n--- STEP 2: PREPROCESSING DATA ---")
    x_train_norm = x_train.astype("float32") / 255.0
    x_test_norm = x_test.astype("float32") / 255.0
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    print("Pixel values normalized to range [0.0, 1.0].")

    return (x_train_norm, y_train_flat), (x_test_norm, y_test_flat)


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """Step 3a: Define a clean, effective Convolutional Neural Network architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 3
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),

        # Classifier Head
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training_history(history):
    """Helper function to save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy subplot
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Loss subplot
    axes[1].plot(history.history["loss"], label="Train Loss", marker="o")
    axes[1].plot(history.history["val_loss"], label="Validation Loss", marker="o")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=150)
    print("Saved training curves plot to 'cnn_training_curves.png'")


def display_sample_predictions(model, x_test, y_test, num_samples=10):
    """Step 5: Show actual vs predicted labels and save visual grid for report."""
    print("\n--- STEP 5: SAMPLE PREDICTIONS ---")
    predictions = model.predict(x_test[:num_samples], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    axes = axes.flatten()

    for i in range(num_samples):
        actual_label = CLASS_NAMES[y_test[i]]
        pred_label = CLASS_NAMES[predicted_classes[i]]
        confidence = np.max(predictions[i]) * 100
        status = "CORRECT" if y_test[i] == predicted_classes[i] else "INCORRECT"

        print(f"Sample {i+1:02d} | Actual: {actual_label:<10} | Predicted: {pred_label:<10} ({confidence:.1f}%) -> {status}")

        axes[i].imshow(x_test[i])
        title_color = "green" if status == "CORRECT" else "red"
        axes[i].set_title(f"Act: {actual_label}\nPred: {pred_label} ({confidence:.0f}%)", color=title_color, fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("cnn_sample_predictions.png", dpi=150)
    print("Saved sample predictions grid to 'cnn_sample_predictions.png'")


def run_part_a_cnn_pipeline():
    """Main workflow executor for Part A (CNN Image Classification)."""
    (x_train, y_train), (x_test, y_test) = load_and_explore_data()

    print("\n--- STEP 3: BUILDING & TRAINING CNN MODEL ---")
    model = build_cnn_model()
    model.summary()

    # Train for 10 epochs (adjust epochs if needed for higher accuracy)
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    plot_training_history(history)

    print("\n--- STEP 4: EVALUATING MODEL ON TEST DATA ---")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")

    display_sample_predictions(model, x_test, y_test, num_samples=10)

    print("\n--- STEP 6: SIMPLE EXPLANATION OF RESULTS ---")
    print(
        f"The CNN achieved {test_acc * 100:.2f}% accuracy on 10,000 unseen test images. "
        "Convolutional layers learned edges, textures, and shapes to classify objects "
        "across 10 categories. Visually similar classes (like Cat vs Dog or Automobile vs Truck) "
        "account for most misclassifications and can be improved with data augmentation and deeper layers."
    )


# ==============================================================================
# PART B: STREAMLIT APPLICATION (STUDENT MARKS & CGPA CALCULATOR)
# ==============================================================================

def calculate_grade(percentage):
    """Helper function to compute letter grade and remarks from percentage."""
    if percentage >= 90:
        return "O (Outstanding)", "Excellent performance! Keep it up."
    elif percentage >= 80:
        return "A+ (Excellent)", "Very strong understanding of subjects."
    elif percentage >= 70:
        return "A (Very Good)", "Good work, with room to push for top scores."
    elif percentage >= 60:
        return "B+ (Good)", "Above average; focus on weak areas."
    elif percentage >= 50:
        return "B (Average)", "Satisfactory pass; consistent revision recommended."
    elif percentage >= 40:
        return "P (Pass)", "Borderline pass; needs dedicated improvement."
    else:
        return "F (Fail)", "Did not meet minimum passing criteria. Re-attempt required."


def run_part_b_streamlit_app():
    """Main UI and logic for Part B (Streamlit Web Application)."""
    import streamlit as st

    st.set_page_config(
        page_title="Student Marks & CGPA Calculator",
        page_icon="🎓",
        layout="centered"
    )

    # Clear Title and Instructions
    st.title("🎓 Student Marks & Percentage Calculator")
    st.markdown(
        "Enter the student's details and subject marks below to calculate total score, "
        "overall percentage, approximate CGPA (10-point scale), and final grade."
    )
    st.divider()

    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Student Full Name", placeholder="e.g., Aarav Sharma")
    with col2:
        num_subjects = st.number_input("Number of Subjects", min_value=1, max_value=10, value=5, step=1)

    max_marks_per_subject = st.number_input(
        "Maximum Marks per Subject", min_value=10, max_value=200, value=100, step=10
    )

    st.subheader("Enter Subject Marks")
    subject_marks = []
    cols = st.columns(2)
    for idx in range(int(num_subjects)):
        with cols[idx % 2]:
            mark = st.number_input(
                f"Subject {idx + 1} Marks",
                min_value=0.0,
                max_value=float(max_marks_per_subject),
                value=75.0,
                step=1.0,
                key=f"subj_{idx}"
            )
            subject_marks.append(mark)

    st.divider()

    # Processing & Validation on Button Click
    if st.button("Calculate Result", type="primary", use_container_width=True):
        # Basic Invalid-Input Handling
        if not student_name.strip():
            st.error("⚠️ Please enter a valid Student Name before calculating.")
            return

        if any(m < 0 or m > max_marks_per_subject for m in subject_marks):
            st.error(f"⚠️ All marks must be between 0 and {max_marks_per_subject}.")
            return

        # Core Calculations
        total_obtained = sum(subject_marks)
        total_maximum = num_subjects * max_marks_per_subject
        percentage = (total_obtained / total_maximum) * 100
        cgpa = round(percentage / 9.5, 2)
        cgpa = min(cgpa, 10.0)  # Cap CGPA at 10.0
        grade, remarks = calculate_grade(percentage)

        # Check individual subject pass/fail (40% threshold)
        pass_threshold = 0.40 * max_marks_per_subject
        failed_subjects = [i + 1 for i, m in enumerate(subject_marks) if m < pass_threshold]

        # Clear Output Display
        st.success(f"📊 Result Summary for **{student_name.strip()}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Marks", f"{total_obtained:.1f} / {total_maximum:.0f}")
        m2.metric("Percentage", f"{percentage:.2f}%")
        m3.metric("Estimated CGPA", f"{cgpa:.2f} / 10")

        st.info(f"**Final Grade:** {grade}  \n**Remarks:** {remarks}")

        if failed_subjects:
            st.warning(
                f"⚠️ Note: Scored below 40% passing mark in Subject(s): "
                f"{', '.join(map(str, failed_subjects))}."
            )
        else:
            st.balloons()


# ==============================================================================
# ENTRY POINT CONTROLLER
# ==============================================================================

if __name__ == "__main__":
    # When executed via `python StudentName_CNN_Streamlit_Project.py --train`, run Part A.
    # When executed via `streamlit run StudentName_CNN_Streamlit_Project.py`, run Part B.
    if "--train" in sys.argv:
        run_part_a_cnn_pipeline()
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                run_part_b_streamlit_app()
            else:
                print("Running Part A (CNN Pipeline) by default...")
                print("Tip: To launch Part B (Streamlit App), run:")
                print("     streamlit run StudentName_CNN_Streamlit_Project.py")
                run_part_a_cnn_pipeline()
        except ImportError:
            run_part_a_cnn_pipeline()
