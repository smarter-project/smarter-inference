# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class AdmissionControllerException(Exception):
    pass


class NoPerfDataFoundException(AdmissionControllerException):
    pass


class DeadlineMissException(AdmissionControllerException):
    pass


class UnsatisfiableRequestException(AdmissionControllerException):
    pass
