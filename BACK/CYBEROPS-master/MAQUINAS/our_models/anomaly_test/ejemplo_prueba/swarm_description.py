# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

SWARM_DESCRIPTION = {
  "includedFields": [
    {
      "fieldName": "timestamp",
      "fieldType": "datetime"
    },
    {
      "fieldName": "value",
      "fieldType": "float",
      "maxValue": 900000000,
      "minValue": 0.0
    }
  ],
  "streamDef": {
    "info": "value",
    "version": 1,
    "streams": [
      {
        "info": "Rec Center",
        "source": "file://ec2_disk_write_bytes_c0d644_labeled.csv",
        "columns": [
          "*"
        ]
      }
    ]
  },

  "inferenceType": "TemporalAnomaly",
  "inferenceArgs": {
    "predictionSteps": [
      1
    ],
    "predictedField": "value"
  },
  "iterationCount": -1,
  "swarmSize": "medium"
}
