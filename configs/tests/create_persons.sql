-- Table: fr.persons

-- DROP TABLE fr.persons;

CREATE TABLE fr.persons
(
  id serial NOT NULL,
  unique_id bigint NOT NULL,
  create_time date NOT NULL,
  person_name character varying(50) NOT NULL,
  person_surname character varying(50) NOT NULL,
  person_secondname character varying(50),
  group_id integer,
  CONSTRAINT persons_pkey PRIMARY KEY (unique_id)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE fr.persons
  OWNER TO face_reco_admin;

