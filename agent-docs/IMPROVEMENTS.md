# Overview
Contains the TODOs that are not entire features, but improvements.
1. Use https://api.central.ballerina.io/2.0/docs/<org>/<name>/<version> and https://api.central.ballerina.io/2.0/registry/packages/<org>/<name>/<version> to get the docs instead of scraping the website
2. Incorporate ballerina.io docs (using the git repo) into the RAG. The important pages are 
    - https://ballerina.io/learn/pre-built-integrations/
    - https://ballerina.io/learn/integration-tutorials/
    - https://ballerina.io/learn/by-example/
    However, you don't have to scrape through the website pages. Just fetch https://github.com/ballerina-platform/ballerina-dev-website and get the needed pages into the DB. Make sure that you preserve the git repo somewhere, so that you dont have to fetch it every time an ingestion happens
3. Add examples from the code repository into the db - The pattern is similar to https://github.com/ballerina-platform/module-ballerinax-kafka/tree/master/examples.
