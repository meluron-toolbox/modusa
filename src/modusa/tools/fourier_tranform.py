#!/usr/bin/env python3


from modusa import excp
from modusa.decorators import validate_args_type
from modusa.tools.base import ModusaTool


class FourierTransform(ModusaTool):
	"""
	
	"""
	
	#--------Meta Information----------
	_name = ""
	_description = ""
	_author_name = "Ankit Anand"
	_author_email = "ankit0.anand0@gmail.com"
	_created_at = "2025-07-11"
	#----------------------------------
	
	def __init__(self):
		super().__init__()
	