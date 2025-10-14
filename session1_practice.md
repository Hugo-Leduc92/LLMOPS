# Session 1 - Practice

Today's goal is to have your local and cloud environments set up and ready to go for the rest of the course. This means:
1. Setting up a Git repository for your team, and a Python virtual environment with the key packages we will be using.
2. Setting up a GCP project for your team, with the key APIs activated.
3. Making sure you can access GCP from your local environment!

Good luck!

## 1. Local set up

### Pre-requisites
Make sure you have the following installed on your machine:
- [Git](https://github.com/git-guides/install-git)
- An IDE of your choice (I personally recommend [VSCode](https://code.visualstudio.com/))
- [Docker](https://docs.docker.com/desktop/)

### Repository and virtual environment
1. With your team members, create a [GitHub]() (or other Git hosting service) repository for the course.
    - You should choose if your repository is public or private.
    - Don't forget to add me (MathieuHS)!

On **one of the team member's machines**, follow these steps to define the Python virtual environment for the team:
1. Clone the repository to your local machine.
    - Pre-requisite: have Git installed on your machine.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) as a package manager.
3. Run `uv init` in the root of your repository to create an initial `pyproject.toml` file and set the Python version.
    - Python version recommended for the course is 3.11.6
4. Create the virtual environment defined by the `pyproject.toml` file with `uv sync`.
    - This will create a `.venv` folder in your repository containing the packages of the virtual environment.
5. Set up your IDE so that it uses the virtual environment you just created.
    - When you open a terminal in your IDE, the `which python` command should point to the `.venv/bin/python` executable.
6. Using `uv add`, install the key GCP packages we will be using in the course:
    - `google-cloud-aiplatform`
    - `google-cloud-bigquery`
    - `google-cloud-storage`
7. Update the `README.md` file in your repository to include, and push your changes to the remote repository.

The **remaining team members** should now:
1. Clone the repository with the updated `README.md` and `pyproject.toml` files.
2. Run `uv sync` to create the virtual environment on their machines based on the existing configuration.
3. Set up their IDE to use the virtual environment!


## 2. Cloud set up

Select a single GCP project for your team to use during the course - the idea is that you share the same project since that's how it often is in real life!

0. Give access to your team members to the selected GCP project.
    - This can be done in the IAM section of the GCP console, other team members should have at least the "Editor" role.
1. Create a billing alert if you don't have one already!
    - Go to the "Budgets & alerts" section of the Billing page in the GCP console.
    - Create a budget for your project, and set an alert at 50% of your budget.
2. Take a tour of the GCP console, especially BigQuery, GCS, and Vertex AI.
3. Create a GCS bucket for the course, and upload the raw Yoda sentences dataset to it.
    - Use a multi-regional EU location for the bucket.
4. Create a BigQuery dataset for the course, and load the Yoda sentences dataset from GCS to a BigQuery table.
    - You can explore the dataset using SQL in the BigQuery console.
5. By default, GCP sets GPU quotas to 0 for custom training (which is what we'll do session 3), meaning you can't access a GPU out-of-the-box! We can submit a request to increase the quota for a specific region and GPU type.
    - To increase your GPU quota, go to the "Quotas" section of the IAM & Admin page in the GCP console.
    - Filter by "Service: Vertex AI", "Region: europe-west2", "Name:Custom model training Nvidia T4 GPUs per region"
    - Click on "Edit Quotas" and fill in the form to request a quota increase to 1.
    - The request usually takes a few hours to be approved.

## 3. Accessing the cloud from your local environment

1. Install locally the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
2. Authenticate, and set the project with the following commands:
    ```bash
    gcloud auth login
    gcloud config set project PROJECT_ID
    ```
3. The owner of the GCP account should enable the VertexAI, Resource manager, BigQuery, and Cloud Storage APIs for the project:
    ```bash
    gcloud services enable aiplatform.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    gcloud services enable bigquery.googleapis.com
    gcloud services enable storage-component.googleapis.com
    ```
4. Create a `.env.example` and a `.env` file that contain key variables for GCP:
    - `GCP_PROJECT_ID`: your GCP project ID
    - `GCP_REGION`: the GCP region you will be using (e.g. `us-central1`)
    - `GCP_BUCKET_NAME`: the name of a GCS bucket you created!
    - Don't forget to add `.env` to your `.gitignore` file!

5. Write a short script to test your GCP setup. It should:
    - Load from environment variables the project ID, region, and bucket name using `os.getenv()`.
    - Check that you can initialize a Vertex AI client with the given project and location using `aiplatform.init` from the `google.cloud.aiplatform` package.
    - Check that you can access the GCS bucket by creating a storage client, and fetching the bucket contents with the `google.cloud.storage` package.
