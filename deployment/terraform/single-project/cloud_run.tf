# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Minimal scale-to-zero Cloud Run v2 configuration serving the unified React UI + ADK API.

variable "cloud_run_image" {
  type        = string
  description = "Container image URI for the unified Cloud Run UI + ADK backend."
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

resource "google_cloud_run_v2_service" "ui_and_agent_api" {
  name     = var.project_name
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = google_service_account.app_sa.email
    max_instance_request_concurrency = 8
    timeout                          = "180s"

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.cloud_run_image

      ports {
        container_port = 8080
      }

      resources {
        cpu_idle = true
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "true"
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = "global"
      }

      env {
        name  = "USE_SECRET_MANAGER"
        value = "true"
      }

      env {
        name  = "GITHUB_TOKEN_SECRET_ID"
        value = google_secret_manager_secret.github_token.secret_id
      }

      env {
        name  = "OTEL_SERVICE_NAME"
        value = var.project_name
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }

  depends_on = [google_project_service.services]
}
