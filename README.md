## API for Festum App
The implementation of Festum API using FastAPI.

### Installation
> git clone https://github.com/Talgin/festum_api.git
- Change settings.py to point to desired service addresses
- Change settings.py to use either cpu or gpu (use_cpu flag)
- Create folders: crops

### Running the service
> docker-compose up -d

### Issues
Sometimes you can encounter bbox errors. One solution can be to:
  - Go to rcnn/cython and do (you have to have Cython package installed):
  > python setup.py build_ext --inplace

### CHANGE HISTORY (started this in 13.05.2023)
- First commits

### TO-DO
- [x] Endpoint to upload mutiple images
- [ ] Change utils.py
- [ ] Functionality to compare photos with stars database
- [ ] Finish FAISS integration for large database (current)
- [ ] Try ScaNN
- [ ] Finish unit-tests
- [ ] Write comments for each function
- [ ] Accept more than maximum requests at one time - think about it
- [ ] Refine code (object reusability, client creation, database connection, configs)
- [ ] Refine face recognition algo - change QMagFace as it may have some errors during deployment
- [ ] Add Metadata and Docs descriptions according to [FastAPI Docs](https://fastapi.tiangolo.com/tutorial/metadata/)
- [ ] Add scaNN search functionality
- [ ] Add license checking at startup - connect to license server during installation to obtain license file or write to license db mac
- [ ] Add docker images to docker hub and update readme
- [ ] List all licenses in one file
- [ ] Connect with MLOps pipeline
- [ ] Create documentation (dev, user)