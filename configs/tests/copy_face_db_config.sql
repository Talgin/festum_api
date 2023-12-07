--
-- PostgreSQL database dump
--

-- Dumped from database version 10.6
-- Dumped by pg_dump version 10.15 (Ubuntu 10.15-0ubuntu0.18.04.1)

-- Started on 2021-12-21 21:42:14 +06

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: fr; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA fr;


ALTER SCHEMA fr OWNER TO face_reco_admin;


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: adminpack; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS adminpack WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION adminpack; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION adminpack IS 'administrative functions for PostgreSQL';


--
-- Name: cube; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS cube WITH SCHEMA public;


--
-- Name: EXTENSION cube; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION cube IS 'data type for multidimensional cubes';


--
-- Name: akati_norm(public.cube); Type: FUNCTION; Schema: fr; Owner: face_reco_admin
--

CREATE FUNCTION fr.akati_norm(vector public.cube) RETURNS double precision
    LANGUAGE plpgsql
    AS $$
declare 
indx integer = 1; -- cube_dim(vector)
sum_of_vec float = 0; 
--i integer = 0; j integer = 1;
begin
if (cube_dim(vector)<512) then
return 0;
end if;

loop
exit when indx = 513;
sum_of_vec = sum_of_vec + cube_ll_coord(vector,indx)*cube_ll_coord(vector,indx);
indx = indx +1;
end loop;
return sum_of_vec;
end;

$$;


ALTER FUNCTION fr.akati_norm(vector public.cube) OWNER TO face_reco_admin;

--
-- Name: akati_sum(public.cube); Type: FUNCTION; Schema: fr; Owner: face_reco_admin
--

CREATE FUNCTION fr.akati_sum(vector public.cube) RETURNS double precision
    LANGUAGE plpgsql
    AS $$
declare 
indx integer = 1; -- cube_dim(vector)
sum_of_vec float = 0; 
--i integer = 0; j integer = 1;
begin
if (cube_dim(vector)<512) then
return 0;
end if;

loop
exit when indx = 513;
sum_of_vec = sum_of_vec + cube_ll_coord(vector,indx);
indx = indx +1;
end loop;
return sum_of_vec;
end;

$$;


ALTER FUNCTION fr.akati_sum(vector public.cube) OWNER TO face_reco_admin;

--
-- Name: cos_dis(public.cube, public.cube); Type: FUNCTION; Schema: fr; Owner: face_reco_admin
--

CREATE FUNCTION fr.cos_dis(vec1 public.cube, vec2 public.cube) RETURNS double precision
    LANGUAGE plpgsql
    AS $$
declare 
indx integer = 1; -- cube_dim(vector)
sum_of_vec float = 0; 
--i integer = 0; j integer = 1;
begin

loop
exit when indx = 513;
sum_of_vec = sum_of_vec + cube_ll_coord(vec1,indx)*cube_ll_coord(vec2,indx);
indx = indx +1;
end loop;
return sum_of_vec;
end;

$$;


ALTER FUNCTION fr.cos_dis(vec1 public.cube, vec2 public.cube) OWNER TO face_reco_admin;

SET default_with_oids = false;

--
-- Name: faces; Type: TABLE; Schema: fr; Owner: face_reco_admin
--

CREATE TABLE fr.faces (
    unique_id bigint,
    vector public.cube
);


ALTER TABLE fr.faces OWNER TO face_reco_admin;

--
-- Name: person_info; Type: TABLE; Schema: fr; Owner: face_reco_admin
--

CREATE TABLE fr.persons (
    id integer NOT NULL,
    unique_id bigint NOT NULL,
    person_name character varying(40) NOT NULL,
    person_surname character varying(40) NOT NULL,
    person_secondname character varying(40),
    insert_date timestamp with time zone,
    group_id integer,
    person_iin character varying(20)
);


ALTER TABLE fr.persons OWNER TO face_reco_admin;

--
-- Name: persons_id_seq; Type: SEQUENCE; Schema: fr; Owner: face_reco_admin
--

CREATE SEQUENCE fr.persons_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE fr.persons_id_seq OWNER TO face_reco_admin;

--
-- Name: persons_id_seq; Type: SEQUENCE OWNED BY; Schema: fr; Owner: face_reco_admin
--

ALTER SEQUENCE fr.persons_id_seq OWNED BY fr.persons.id;

--
-- Name: vectors_archive; Type: TABLE; Schema: fr; Owner: face_reco_admin
--

CREATE TABLE fr.vectors_archive (
    id integer NOT NULL,
    unique_id bigint NOT NULL,
    vector public.cube,
    camera_id character varying(40),
    server_id integer
);


ALTER TABLE fr.vectors_archive OWNER TO face_reco_admin;

--
-- Name: vectors_archive_id_seq; Type: SEQUENCE; Schema: fr; Owner: face_reco_admin
--

CREATE SEQUENCE fr.vectors_archive_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE fr.vectors_archive_id_seq OWNER TO face_reco_admin;

--
-- Name: vectors_archive_id_seq; Type: SEQUENCE OWNED BY; Schema: fr; Owner: face_reco_admin
--

ALTER SEQUENCE fr.vectors_archive_id_seq OWNED BY fr.vectors_archive.id;

--
-- Name: persons id; Type: DEFAULT; Schema: fr; Owner: face_reco_admin
--

ALTER TABLE ONLY fr.persons ALTER COLUMN id SET DEFAULT nextval('fr.persons_id_seq'::regclass);

--
-- Name: vectors_archive id; Type: DEFAULT; Schema: fr; Owner: face_reco_admin
--

ALTER TABLE ONLY fr.vectors_archive ALTER COLUMN id SET DEFAULT nextval('fr.vectors_archive_id_seq'::regclass);

--
-- Name: persons persons_pkey; Type: CONSTRAINT; Schema: fr; Owner: face_reco_admin
--

ALTER TABLE ONLY fr.persons
    ADD CONSTRAINT persons_pkey PRIMARY KEY (id);

--
-- Name: faces faces_pkey; Type: CONSTRAINT; Schema: fr; Owner: face_reco_admin
--

ALTER TABLE ONLY fr.faces
    ADD CONSTRAINT faces_pkey PRIMARY KEY (unique_id);

--
-- Name: vectors_archive vectors_archive_pkey; Type: CONSTRAINT; Schema: fr; Owner: face_reco_admin
--

ALTER TABLE ONLY fr.vectors_archive
    ADD CONSTRAINT vectors_archive_pkey PRIMARY KEY (id);

--
-- Name: persons_unique_id_4570c12e; Type: INDEX; Schema: fr; Owner: face_reco_admin
--

CREATE INDEX persons_unique_id_4570c12e ON fr.persons USING btree (unique_id);

--
-- Name: faces_unique_id_4575c15e; Type: INDEX; Schema: fr; Owner: face_reco_admin
--

CREATE INDEX faces_unique_id_4575c15e ON fr.faces USING btree (unique_id);

--
-- Name: vectors_archive_unique_id_4574c14e; Type: INDEX; Schema: fr; Owner: face_reco_admin
--

CREATE INDEX vectors_archive_unique_id_4574c14e ON fr.vectors_archive USING btree (unique_id);


-- Completed on 2022-12-21 21:42:14 +06

--
-- PostgreSQL database dump complete
--


