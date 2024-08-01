# function to pick the single main label
def get_main_label(labels):
    if len(labels) == 1:
      return labels[0]
    elif "UNSURE" in labels:
      return "UNSURE"
    elif None in labels: # this may be something to double check
      return "UNSURE"
    elif labels == None:
      return "UNSURE"
    elif "MIXED" in labels:
      return "MIXED"
    elif len(labels) > 1: # this is when multiple labels are used but not the mixed label
      return "MIXED"
    raise Exception("No main label found")
