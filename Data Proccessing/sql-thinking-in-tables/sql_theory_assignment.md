1. * As far as i have learned databases are base of any ai models , on which they are being trained . In realworld ai system , database stores the large amount of complex data which will help the ai model to be trained properly , so that it is capable of solving the realworld problems. 
   * The type of data which is stored in those databases is structured  data , which is in tabular form . The data should be acccurately processed and cleaned to be used as database for ai models.
   * Structured storage is naccesry , because if the storage of data is not structured then it may led to confusion in model while prediction . It will led to condition like underfitting or overfitting , which is not good sign in training models .

2. * Tables (The Big Boxes)
     A table is just a category. Its like a folder for one "thing." If you have a store, you have a table for Stuff_To_Sell and a table for People_Who_Buy. You dont want to mix them up or it gets real confusing real fast.

    * Columns (The Top Part)
     Columns are the labels at the top. They tell you what kind of info goes there. Like Name, Price, or Phone_Numbr. You cant put a name in the price spot because the computer gets mad. Its like a rule for what fits in the box.

    * Rows (The Peeple/Things)
    A row is just one single item. If you have a Students table, one row is just ME. It has my name, my grade (probly a D), and my ID. Every row has to be unique so the database doesnt get confused between two people with the same name.

    > why one thing per tabel? : --> If we have diffrent entitties in a single table , it would eb hardd for us to modify and understaand that table . for example : if we have employee and orders data in one table then for changing one entities , we will  have to mess up with other entity as well . which would be tidious task .

3. A primary key is a column (or set of columns) that uniquely identifies each row in a table.

   * why it gotta be unique?
   if two rows have the same primary key, the whole thing breaks.

    no twins: if u have two "ID 10" rows, and u tell the computer "delete ID 10," it might delete both of them or the wrong one.

    finding stuff: it's like a shortcut. if the key is unique, the computer can find exactly what u want super fast without lookin at every single name or address.

   * what about the "non-null" thing?
   "non-null" just means it cant be empty. u cant have a row with no ID.

   the anchor: if a row doesnt have a primary key, it’s like a ghost. u cant link it to other tabels and u cant really find it easily.

   rules: the database basically forces u to put sumthin there so the data doesnt get lost or "homeless" i guess.

   * how it helps identify records
   without a primary key, a tabel is just a big pile of info. with it, every row has a "name" (the ID) that never changes. so even if a customer changes their email, their phone number, and their last name, their customer_id stays the same. that way the database always knows its still the same person.

4. What Information Does a Schema Define?
A schema doesn't hold the data itself; it defines the rules and shapes that the data must follow. It typically specifies:

   Tables: The primary containers (e.g., Users, Orders, Products).

   Fields (Columns): The specific attributes within a table (e.g., Email, Price, Date).

   Data Types: What kind of data is allowed in each field? For example, an Age column must be an integer, while a Username must be a string of text.

   Constraints: Rules like "this field cannot be empty" (NOT NULL) or "this value must be unique" (UNIQUE).

   Relationships: How tables connect to one another. For instance, a "User ID" in the Orders table must match a "User ID" in the Users table (Primary and Foreign Keys).

   Why are Schemas Important?
   Without a schema, a database would eventually become a "data swamp"—a messy pile of information that is difficult to search or trust. Here is why they are vital:

   1. Data Integrity and Consistency
   The schema acts as a gatekeeper. If someone tries to enter a "Phone Number" into a field meant for "Currency," the schema rejects it. This ensures that every piece of data in the system follows the same format, preventing errors downstream in apps or reports.

   2. Scalability and Organization
   As a project grows, having a clear schema allows developers to understand exactly how data is organized without having to guess. It makes it much easier to add new features or tables without breaking existing ones.

   3. Efficiency (Performance)
   A well-designed schema helps the database engine find information faster. By defining Indexes within the schema, you're essentially creating a "Table of Contents" for your data, allowing the system to skip the "read every single row" process.

   4. Security
   Schemas allow administrators to control access. You can give a user permission to see the Products table but completely hide the CreditCardInfo table, all based on the schema's structure.
5. 
   Relational databases use Primary Keys (unique IDs) and Foreign Keys to link tables. In a Users and Orders system, each user has a unique user_id (Primary Key). To connect them, the Orders table includes a user_id column as a Foreign Key.

   This creates a One-to-Many relationship: one user can place multiple orders, but each order points back to exactly one user. This structure ensures referential integrity, meaning you can’t have an order for a user who doesn't exist, keeping your data clean and organized.
