from pymilvus import CollectionSchema, FieldSchema, DataType

# We're going to create a collection with 5 fields.
# +-+----------------+------------+------------------+------------------------------+
# | |   field name   | field type | other attributes |       field description      |
# +-+----------------+------------+------------------+------------------------------+
# |1|  "person_id"   |   VarChar  |  is_primary=True |      "primary field"         |
# | |                |            |   auto_id=False  |                              |
# +-+----------------+------------+------------------+------------------------------+
# |2|    "random"    |    Double  |                  |      "a double field"        |
# +-+----------------+------------+------------------+------------------------------+
# |3|  "embeddings"  | FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+----------------+------------+------------------+------------------------------+
person_id = FieldSchema(
  name="person_id",
  dtype=DataType.INT64,
  is_primary=True,
)
person_name = FieldSchema(
  name="person_name",
  dtype=DataType.VARCHAR,
  max_length=200,
  # The default value will be used if this field is left empty during data inserts or upserts.
  # The data type of `default_value` must be the same as that specified in `dtype`.
  default_value="Unknown"
)
person_surname = FieldSchema(
  name="person_surname",
  dtype=DataType.VARCHAR,
  max_length=200,
  # The default value will be used if this field is left empty during data inserts or upserts.
  # The data type of `default_value` must be the same as that specified in `dtype`.
  default_value="Unknown"
)
person_middlename = FieldSchema(
  name="person_middlename",
  dtype=DataType.VARCHAR,
  max_length=200,
  # The default value will be used if this field is left empty during data inserts or upserts.
  # The data type of `default_value` must be the same as that specified in `dtype`.
  default_value="Unknown"
)
person_feature = FieldSchema(
  name="person_feature",
  dtype=DataType.FLOAT_VECTOR,
  dim=512
)

fields = [person_id, person_name, person_surname, person_middlename, person_feature]