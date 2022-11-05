# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .server_local import NginxServerLocal


class NginxServerFactory:
    """
    A factory for creating Nginx instances
    """

    @staticmethod
    def create_server_local(path, config, log_path=None):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the nginx executable
        config : NginxServerConfig
            the config object containing arguments for this server instance
        log_path: str
            Absolute path to the triton log file

        Returns
        -------
        NginxServerLocal
        """

        return NginxServerLocal(path=path, config=config, log_path=log_path)
