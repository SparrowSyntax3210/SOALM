def predict_point(
    model,
    encoder,
    FEATURES,
    weather,
    latitude,
    longitude
):

    from .features import build_spatial_features

    X, feature_values = build_spatial_features(
        weather,
        latitude,
        longitude,
        FEATURES
    )

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    risk = encoder.inverse_transform(
        [prediction]
    )[0]

    probability_map = {

        str(encoder.classes_[i]):
        float(probabilities[i])

        for i in range(
            len(encoder.classes_)
        )
    }

    confidence = float(
        max(probabilities)
    )

    return {

        "latitude": float(latitude),

        "longitude": float(longitude),

        "risk": str(risk),

        "probability":
            round(
                float(
                    probability_map[risk]
                ),
                4
            ),

        "confidence":
            round(
                confidence,
                4
            ),

        "class_probabilities":
            probability_map,

        "features":
            feature_values
    }