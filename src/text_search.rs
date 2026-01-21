use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};
use std::sync::{Arc, Mutex};
use anyhow::Result;
use crate::model::Item;

pub struct TextSearch {
    index: Index,
    reader: IndexReader,
    writer: Arc<Mutex<IndexWriter>>,
    fields: SchemaFields,
}

#[derive(Clone)]
struct SchemaFields {
    id: Field,
    title: Field,
    category: Field,
}

impl TextSearch {
    pub fn new(index_path: &str) -> Result<Self> {
        let mut schema_builder = Schema::builder();
        
        let id = schema_builder.add_u64_field("id", STORED);
        let title = schema_builder.add_text_field("title", TEXT | STORED);
        let category = schema_builder.add_text_field("category", STRING | STORED);
        
        let schema = schema_builder.build();
        let fields = SchemaFields { id, title, category };

        std::fs::create_dir_all(index_path)?;
        
        let index = Index::open_or_create(
            tantivy::directory::MmapDirectory::open(index_path)?,
            schema.clone()
        )?;
        let writer = index.writer(50_000_000)?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            writer: Arc::new(Mutex::new(writer)),
            fields,
        })
    }

    pub fn index_item(&self, item: &Item) -> Result<()> {
        let mut writer = self.writer.lock().map_err(|_| anyhow::anyhow!("Poisoned lock"))?;
        
        let doc = doc!(
            self.fields.id => item.id as u64,
            self.fields.title => item.name.clone(),
            self.fields.category => item.category.clone()
        );
        
        writer.add_document(doc)?;
        Ok(())
    }

    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock().map_err(|_| anyhow::anyhow!("Poisoned lock"))?;
        writer.commit()?;
        Ok(())
    }

    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<u32>> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.title]);
        
        let query = query_parser.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
        
        let mut results = Vec::new();
        for (_score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(id_val) = retrieved_doc.get_first(self.fields.id) {
                if let Some(id) = id_val.as_u64() {
                    results.push(id as u32);
                }
            }
        }
        
        Ok(results)
    }
}
