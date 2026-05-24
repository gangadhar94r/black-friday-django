"""
Views for the predictor app.
Handles user requests and returns responses.
"""
from django.shortcuts import render
from . import ml_handler


# Data to pass to the prediction form template
FORM_OPTIONS = {
    'occupation_range': list(range(21)),      # 0, 1, 2, ..., 20
    'product_cat_range': list(range(1, 19)),  # 1, 2, 3, ..., 18
}


def home(request):
    """Renders the home page."""
    return render(request, 'predictor/home.html')


def predict(request):
    """
    Handles the prediction form page.
    GET: Shows the empty form.
    POST: Runs ML prediction and shows the result.
    """
    prediction = None
    form_data = {}
    error = None

    if request.method == 'POST':
        form_data = request.POST.dict()
        model_choice = form_data.get('model_choice', 'random_forest')

        try:
            prediction = ml_handler.predict(form_data, model_choice=model_choice)
        except Exception as e:
            error = f"Prediction error: {str(e)}"

    context = {
        'form_data': form_data,
        'prediction': prediction,
        'error': error,
        **FORM_OPTIONS,
    }
    return render(request, 'predictor/predict.html', context)
def analysis(request):
    """
    Shows model performance metrics and feature importance.
    Loads the metrics.json file created during training.
    """
    metrics = ml_handler._load_models()['metrics']

    # Get feature importance from Random Forest
    feature_importance = metrics['random_forest']['feature_importance']

    # Sort features by importance (most important first)
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return render(request, 'predictor/analysis.html', {
        'metrics': metrics,
        'sorted_features': sorted_features,
    })
def about(request):
    """Renders the About page with project information."""
    return render(request, 'predictor/about.html')