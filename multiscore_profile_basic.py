#######################
# Document Explorerer
######################

# step 1: embed the comparison_doc
# step 2: embed each text
# step 3: get scores
# step 4: evaluates if score is succss or fail
# step 5: if success: do stuff with text

import json
from datetime import datetime
start_time_whole_single_task = datetime.now()

# preset/reset
article_data = []

##########################################
# Make comparison doc and vectorize it
##########################################
comparison_doc = """
words words worlds
"""

embedding1 = get_vector(comparison_doc)


article_id_counter = 0

for this_index, article in enumerate(list_of_texts):

  
    extracted_article_string = article


    ############################
    # Do embedding search here:
    ############################

    embedding2 = get_vector(extracted_article_string)

    
    ##################################
    # Do basic embedding search here:
    ##################################

    list_of_comparison_function_tuples = [
        (cosine_similarity_distance, "cosine_similarity_distance"),
        (correlation_distance_dissimilarity_measure, "correlation_distance_dissimilarity_measure"),
        (pearson_correlation, "pearson_correlation"),
        (canberra_distance, "canberra_distance"),
        (euclidean_distance, "euclidean_distance"),
        (manhattan_distance, "manhattan_distance"),
        (minkowski_distance, "minkowski_distance"),
        (squared_euclidean_distance_dissimilarity_measure, "squared_euclidean_distance_dissimilarity_measure"),
        (chebyshev_distance, "chebyshev_distance"),
        (kendalls_rank_correlation, "kendalls_rank_correlation"),
        (bray_curtis_distance_dissimilarity, "bray_curtis_distance_dissimilarity"),
        (normalized_dot_product, "normalized_dot_product"),
        (spearmans_rank_correlation, "spearmans_rank_correlation"),
        (total_variation_distance_dissimilarity_measure, "total_variation_distance_dissimilarity_measure"),
    ]


    # Arguments to pass to the functions
    arguments = (embedding1, embedding2, True)

    # print(f"For {comparison_doc} vs. {extracted_article_string}")

    list_of_boolean_scores = []

    """
    compare to results of keyword search

    do self-search
    
    do paraphrase search
    
    Score_Profile
    1. get a boolean
    2. get threshold
    3. get distance past threshold
    4. get weak, medium, strong distance score



    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }
    
    """
    passed_metrics = []
    failed_metrics = []
    pass_fail_list = []
    list_of_profiles = []
    counter = 0
    
    # Iterate through the functions and call each one with the arguments
    for this_function_tuple in list_of_comparison_function_tuples:
        function_pointer = this_function_tuple[0]
        
        result_profile = function_pointer(*arguments)    

        print(f"result_profile {result_profile}")
        
        boolean_score = result_profile['boolean']

        """
        Look at which scores are pass or fail
        """
        # preset/reset
        passed_metrics = []
        failed_metrics = []

        if boolean_score:
            passed_metrics.append(counter)

        else:
            failed_metrics.append(counter)

        # print(raw_score)
        list_of_boolean_scores.append(boolean_score)

        list_of_profiles.append(result_profile)

        counter += 1

    pass_fail_list.append( (counter, passed_metrics,failed_metrics)  )
    
    ratio_score = list_of_boolean_scores.count(True)

    print(f"{ratio_score} / {len(list_of_boolean_scores)}")

    # input("PointBreak")

    decimal_percent_true = ratio_score / len(list_of_boolean_scores)
    
    # target_score_decimal_percent = 0.5
    target_score_decimal_percent = 5 / len(list_of_boolean_scores)

    if decimal_percent_true >= target_score_decimal_percent:

        # Append the data to the list
        article_data.append({
            'article_id': article_id_counter,
            'scores': f"{ratio_score} / {len(list_of_boolean_scores)}",
            'pass_fail_list': pass_fail_list,
            'list_of_profiles': list_of_profiles,
            
            

        })
        
    article_id_counter += 1


    # Check if the abstract contains any of the keywords
    # TODO

#############
# Write Data
#############

# Posix UTC Seconds
# make readable time
from datetime import datetime, UTC
date_time = datetime.now(UTC)
clean_timestamp = date_time.strftime('%Y-%m-%d__%H%M%S%f')


# Save the data to a JSON file
with open(f'articles_{clean_timestamp}.json', 'w') as f:
    json.dump(article_data, f)


# Create an HTML file
html = '<html><body>'
for article in article_data:
    html += f'<h2><a href="scores">{article["scores"]}</a></h2>'
    html += f'<p>{article["article_id"]}</p>'
    html += f'<p>{article["pass_fail_list"]}</p>'
    html += f'<p>{article["list_of_profiles"]}</p>'

html += '</body></html>'


# Save the HTML to a file
with open(f'articles{clean_timestamp}.html', 'w') as f:
    f.write(html)



# Duration time
end_time_whole_single_task = datetime.now()
duration_time = duration_min_sec(start_time_whole_single_task, end_time_whole_single_task)


# Prints
print(f'Made: articles{clean_timestamp}.html')

for this in article_data:
        print(this['scores'])

print(f"duration_time {duration_time}")
