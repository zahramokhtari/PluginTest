import os
import re
import random
from xml.etree import ElementTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), 'pom1.xml')
with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
    data = f.read().split("\n----------------------------------------------\n")

# Remove any text or whitespace before the XML declaration
for i, pom in enumerate(data):
    match = re.search(r"<\?xml", pom)
    if match is not None:
        data[i] = pom[match.start():]

# Preprocess the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

# Train the k-nearest neighbors model
knn_model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
knn_model.fit(X)


def recommend(test_pom):
    # Parse the user's pom.xml file and extract the list of dependencies
    root = ElementTree.fromstring(test_pom)
    dependencies = []
    for dependency in root.iter("{http://maven.apache.org/POM/4.0.0}dependency"):
        group_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}groupId")
        artifact_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}artifactId")
        if group_id_elem is not None and artifact_id_elem is not None:
            coordinates = group_id_elem.text + ":" + artifact_id_elem.text
            dependencies.append(coordinates)

    # Preprocess the user's pom.xml file
    user_text = " ".join(dependencies)
    user_vec = vectorizer.transform([user_text])

    # Find the k-nearest neighbors to the user's pom.xml file
    distances, indices = knn_model.kneighbors(user_vec)

    # Get the indices of the k-nearest neighbors
    indices = indices[0]

    # Get the top 3 most similar pom.xml files
    top_poms = [data[j] for j in indices[:2]]

    # Extract the list of libraries used in the top 3 most similar pom.xml files
    libraries = set()
    for top_pom in top_poms:
        root = ElementTree.fromstring(top_pom)
        for dependency in root.iter("{http://maven.apache.org/POM/4.0.0}dependency"):
            group_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}groupId")
            artifact_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}artifactId")
            if group_id_elem is not None and artifact_id_elem is not None:
                coordinates = group_id_elem.text + ":" + artifact_id_elem.text
                libraries.add(coordinates)

    # Remove the libraries that are already used in the user's pom.xml file
    for dependency in dependencies:
        if dependency in libraries:
            libraries.remove(dependency)

    # Return the recommended libraries
    return list(libraries) if libraries else []


def modify_dependencies(xml_text, deletion_rate=0.5):
    # Parse the XML text
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return 2

    # Find the <dependencies> element using the namespace
    ns = {"ns0": "http://maven.apache.org/POM/4.0.0"}
    dependencies_element = root.find(".//ns0:dependencies", ns)
    if dependencies_element is None:
        # Return the XML unchanged if <dependencies> element is not found
        return 2

    dependencies = dependencies_element.findall(".//ns0:dependency", ns)

    # Calculate the number of dependencies to delete based on the deletion rate
    num_dependencies_to_delete = int(len(dependencies) * deletion_rate)
    if num_dependencies_to_delete < 2:
        return 2

    # Randomly shuffle the dependencies for deletion
    random.shuffle(dependencies)

    # Get the coordinates of deleted dependencies
    deleted_coordinates = []
    for dependency in dependencies[:num_dependencies_to_delete]:
        group_id = dependency.find(".//ns0:groupId", ns).text
        artifact_id = dependency.find(".//ns0:artifactId", ns).text
        deleted_coordinates.append(f"{group_id}:{artifact_id}")
        dependencies_element.remove(dependency)

    # Convert the modified XML back to a string without the ns0: namespace prefix
    ElementTree.register_namespace('', "http://maven.apache.org/POM/4.0.0")
    output = ElementTree.tostring(root, encoding="unicode")
    output = output.replace("ns0:", "")

    # Call the recommend function on the modified XML
    recommended_libraries = recommend(output)

    # Compare the recommended libraries with the deleted coordinates
    num_correct_predictions = sum(1 for lib in recommended_libraries if lib in deleted_coordinates)

    # Calculate and return the accuracy
    return num_correct_predictions / num_dependencies_to_delete


def measure_accuracy():
    num_trials = 0
    total_accuracy = 0.0  # Initialize as a float
    for a in range(500):
        # Randomly select one pom.xml from the dataset
        random_data = random.choice(data)

        # Randomly modify the selected pom.xml and pass it to the recommendation function
        modified_data = modify_dependencies(random_data, deletion_rate=0.2)

        # Update the total accuracy
        if modified_data != 2:
            total_accuracy += modified_data
            num_trials = num_trials + 1

    avg_accuracy = (total_accuracy / num_trials) * 100
    print(f"Average Accuracy: {avg_accuracy:.2f}%")


if __name__ == "__main__":
    measure_accuracy()
