# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from jinja2 import Template

from ..app.constants import NO_NGINX_RATELIMIT
from ..app.model_state import ModelState


class NginxServerConfig:
    """
    A config class to set arguments to the Nginx
    Server in front of Triton Server. An argument set to None will use the server default.
    """

    nginx_conf_template = """
    
worker_processes auto;
pid /run/nginx.pid;
daemon off;
events {
        worker_connections 768;
}
http {
        #### BEGIN: Stuff from the default nginx.conf ####
        tcp_nopush on;
        tcp_nodelay on;
        keepalive_timeout 70;  # Bumped from 65
        types_hash_max_size 2048;
        include /etc/nginx/mime.types;
        default_type application/octet-stream;
        ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
        ssl_prefer_server_ciphers on;

        {% if nginx_logging %}
        error_log /var/log/nginx/error.log;
        access_log on;
        {% else %}
        error_log   /dev/null   crit;
        access_log off;
        {% endif %}
        
        #### END Stuff from the default nginx.conf ####
        gzip on;
        
        # Enable rate limiting
        {% for model_name,model_state in model_states.items() %}
         {% for uuid,load_request in model_state.load_requests.items() %}
         limit_req_zone $binary_remote_addr zone={{ uuid }}_limit:4m rate={{ (load_request.perf_targets.constraints.perf_throughput * 1.1 * 60 ) | int }}r/m;
         {% endfor %}
        {% endfor %}

        server {
                listen 2521 default_server;
                listen [::]:2521 default_server;
                server_name tritonserver;

                # Health check
                location = /health {
                        access_log off;
                        add_header 'Content-Type' 'application/json';
                        return 200 '{"status":"UP"}';
                }
                
                # Reverse-proxy to tritonserver
                location / {
                        proxy_pass        http://localhost:8000;
                        client_max_body_size 100M;
                }

                {% if triton_configs %}

                {% for model_name,triton_config in triton_configs.items() %}
                {% for uuid,load_request in model_states[model_name].load_requests.items() %}
                location /v2/models/{{ uuid }}/ready {
                        proxy_pass        http://localhost:{{ triton_config['http-port'] }}/v2/models/{{ model_name }}/ready;
                        client_max_body_size 100M;
                }

                location /v2/models/{{ uuid }}/infer {
                        proxy_pass        http://localhost:{{ triton_config['http-port'] }}/v2/models/{{ model_name }}/infer;
                        {% if rate_limit %}
                        limit_req         zone={{ uuid }}_limit burst=20;
                        {% endif %}
                        client_max_body_size {{ (model_states[model_name].max_input_size * 1.05) | int }};
                        client_body_buffer_size {{ (model_states[model_name].max_input_size * 1.05) | int }};
                }
                {% endfor %}
                {% endfor %}

                {% else %}

                {% for model_name,model_state in model_states.items() %}
                {% for uuid,load_request in model_state.load_requests.items() %}
                location /v2/models/{{ uuid }}/ready {
                        proxy_pass        http://localhost:8000/v2/models/{{ model_name }}/ready;
                        client_max_body_size 100M;
                }

                location /v2/models/{{ uuid }}/infer {
                        proxy_pass        http://localhost:8000/v2/models/{{ model_name }}/infer;
                        {% if rate_limit %}
                        limit_req         zone={{ uuid }}_limit burst=20;
                        {% endif %}
                        client_max_body_size {{ (model_state.max_input_size * 1.05) | int }};
                        client_body_buffer_size {{ (model_state.max_input_size * 1.05) | int }};
                }
                {% endfor %}
                {% endfor %}

                {% endif %}
        }

        server {
                listen 2522 http2 default_server;
                listen [::]:2522 http2 default_server;
                server_name tritonservergrpc;
                proxy_buffering off;

                # Reverse-proxy to tritonserver
                location /inference.GRPCInferenceService {
                        grpc_pass         grpc://inference_service;
                }

                location /inference.GRPCInferenceService/ModelInfer {
                        grpc_pass         grpc://inference_service;
                        {% for model_name,model_state in model_states.items() %}
                        {% for uuid,load_request in model_state.load_requests.items() %}
                        {% if rate_limit %}
                        limit_req         zone={{ uuid }}_limit burst=20;
                        {% endif %}
                        {% endfor %}
                        {% endfor %}
                        client_max_body_size 100M;
                }
        }

        # Backend gRPC servers
        #
        upstream inference_service {
                zone inference_service 64k;
                server localhost:8001;
        }
}
        """

    def __init__(
        self,
        config_path,
        model_states={},
        nginx_logging=False,
        triton_configs: List[Dict] = None,
    ):
        """
        Construct NginxServerConfig
        """

        # TODO: pass input tensor byte size in the constraints for use
        self._model_states: Dict[str, ModelState] = model_states
        self._config_path = config_path
        self._current_config_text = ""
        self._nginx_logging = nginx_logging
        self._rate_limit = not (NO_NGINX_RATELIMIT)
        self._triton_configs = triton_configs

    def update(self):
        """
        Utility function to convert a config into an nginx config and
        write out to nginx config file
        """

        template = Template(
            self.nginx_conf_template,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self._current_config_text = template.render(
            nginx_logging=self._nginx_logging,
            model_states=self._model_states,
            rate_limit=self._rate_limit,
            triton_configs=self._triton_configs,
        )

    def get_config_path(self):
        return self._config_path

    def get_config(self):
        return self._current_config_text

    def get_model_states(self):
        return self.model_states

    @property
    def model_states(self):
        return self._model_states

    @model_states.setter
    def model_states(self, model_states: Dict[str, ModelState]):
        self._model_states = model_states

    @property
    def triton_configs(self):
        return self._triton_configs

    @triton_configs.setter
    def triton_configs(self, triton_configs: List[Dict]):
        self._triton_configs = triton_configs
