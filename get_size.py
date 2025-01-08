SIZE_CHART = {"XS": 41, "S": 44, "M": 46, "L": 49, "XL": 52, "XXL": 55}


def suggest_size(shoulder_width):
    for size, shoulder_breadth in SIZE_CHART.items():
        if(shoulder_width <= shoulder_breadth):
            return size